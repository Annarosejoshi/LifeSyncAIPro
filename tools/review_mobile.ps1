$ErrorActionPreference = 'Stop'
$reviewRoot = Split-Path -Parent $PSScriptRoot
$reviewOutput = Join-Path $reviewRoot 'artifacts\ui-review'
$reviewChrome = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
$reviewProfile = Join-Path $reviewOutput 'chrome-mobile-profile'
$reviewArgs = @('--headless', '--disable-gpu', '--no-first-run', '--remote-debugging-port=9237', "--user-data-dir=`"$reviewProfile`"", 'about:blank')
$reviewProcess = Start-Process -FilePath $reviewChrome -ArgumentList $reviewArgs -WindowStyle Hidden -PassThru
$reviewSocket = [System.Net.WebSockets.ClientWebSocket]::new()
$script:reviewMessageId = 0

function Send-ReviewCommand($method, $parameters) {
    $script:reviewMessageId++
    $command = @{id=$script:reviewMessageId; method=$method; params=$parameters} | ConvertTo-Json -Depth 12 -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($command)
    $reviewSocket.SendAsync([ArraySegment[byte]]::new($bytes), [System.Net.WebSockets.WebSocketMessageType]::Text, $true, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
    do {
        $stream = [System.IO.MemoryStream]::new()
        do {
            $buffer = New-Object byte[] 65536
            $received = $reviewSocket.ReceiveAsync([ArraySegment[byte]]::new($buffer), [Threading.CancellationToken]::None).GetAwaiter().GetResult()
            $stream.Write($buffer, 0, $received.Count)
        } until ($received.EndOfMessage)
        $message = [System.Text.Encoding]::UTF8.GetString($stream.ToArray()) | ConvertFrom-Json
        $stream.Dispose()
    } until ($message.id -eq $script:reviewMessageId)
    if ($message.error) { throw ($message.error | ConvertTo-Json -Compress) }
    return $message.result
}

try {
    $reviewPages = $null
    for ($attempt=0; $attempt -lt 20; $attempt++) {
        try { $reviewPages = Invoke-RestMethod 'http://127.0.0.1:9237/json'; break } catch { Start-Sleep -Milliseconds 250 }
    }
    $reviewPage = $reviewPages | Where-Object type -eq 'page' | Select-Object -First 1
    $null = $reviewSocket.ConnectAsync([Uri]$reviewPage.webSocketDebuggerUrl, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
    $null = Send-ReviewCommand 'Emulation.setDeviceMetricsOverride' @{width=390; height=844; deviceScaleFactor=1; mobile=$true}
    foreach ($name in @('predictions','checkin','dashboard','wellness','bmi','mental','profile','medication')) {
        $reviewFile = [Uri]::new((Join-Path $reviewOutput "$name.html"))
        $null = Send-ReviewCommand 'Page.navigate' @{url=$reviewFile.AbsoluteUri}
        $null = Send-ReviewCommand 'Runtime.evaluate' @{expression='new Promise(resolve => setTimeout(resolve, 1200))'; awaitPromise=$true}
        $metrics = Send-ReviewCommand 'Runtime.evaluate' @{expression='JSON.stringify({viewport:innerWidth,scroll:document.documentElement.scrollWidth})'; returnByValue=$true}
        Write-Output "$name $($metrics.result.value)"
        $interaction = @'
(() => {
 const results = {errors:window.reviewErrors || []};
 const filter=document.querySelector('[data-condition-filter="1"]');
 if(filter){filter.click();results.filter=[...document.querySelectorAll('[data-condition]')].filter(x=>!x.hidden).length===1;document.querySelector('[data-condition-filter="all"]').click();}
 const launch=document.querySelector('[data-open-chat]');
 if(launch){launch.click();results.chatOpen=!document.getElementById('wellnessChatPanel').hidden;document.getElementById('closeWellnessChat').click();results.chatClose=document.getElementById('wellnessChatPanel').hidden;}
 if(typeof openSidebar==='function'){openSidebar();results.sidebarOpen=document.getElementById('lifesyncSidebar').classList.contains('open');closeSidebar();}
 return JSON.stringify(results);
})()
'@
        $interactionResult = Send-ReviewCommand 'Runtime.evaluate' @{expression=$interaction;returnByValue=$true}
        Write-Output "Interactions $name $($interactionResult.result.value)"
        $capture = Send-ReviewCommand 'Page.captureScreenshot' @{format='png'; captureBeyondViewport=$false}
        [IO.File]::WriteAllBytes((Join-Path $reviewOutput "$name-mobile.png"), [Convert]::FromBase64String($capture.data))
    }
} finally {
    $reviewSocket.Dispose()
    if ($reviewProcess -and -not $reviewProcess.HasExited) { Stop-Process -Id $reviewProcess.Id }
}
