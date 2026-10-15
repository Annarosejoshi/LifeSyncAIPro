function openSidebar(){const sidebar=document.getElementById('lifesyncSidebar');sidebar?.classList.add('open');const overlay=document.getElementById('sidebarOverlay');if(overlay){overlay.style.display='block';overlay.style.pointerEvents='auto';}}
if (typeof Chart !== 'undefined') {
    Chart.defaults.font.size = 16;
    Chart.defaults.font.family = 'Arial, Helvetica, sans-serif';
    for (const chart of Object.values(Chart.instances)) {
        for (const scale of Object.values(chart.options.scales || {})) {
            scale.ticks ||= {};
            scale.ticks.font = {...(scale.ticks.font || {}), size: 16};
            if (scale.title) scale.title.font = {...(scale.title.font || {}), size: 16};
        }
        chart.update('none');
    }
}
function closeSidebar(){document.getElementById('lifesyncSidebar')?.classList.remove('open');const overlay=document.getElementById('sidebarOverlay');if(overlay){overlay.style.display='none';overlay.style.pointerEvents='none';}}
document.getElementById('sidebarOverlay')?.addEventListener('click',closeSidebar);
document.addEventListener('keydown',event=>{if(event.key==='Escape')closeSidebar();});
for(const link of document.querySelectorAll('.sidebar-link'))if(new URL(link.href).pathname===location.pathname){link.classList.add('active');link.setAttribute('aria-current','page');}
for(const button of document.querySelectorAll('[data-coach-prompt]'))button.addEventListener('click',()=>{const input=document.querySelector('[name="coach_question"]');if(input){input.value=button.dataset.coachPrompt;input.focus();input.scrollIntoView({behavior:'smooth',block:'center'});}});
for(const form of document.querySelectorAll('form:not(#wellnessChatForm)'))form.addEventListener('submit',()=>{if(!form.checkValidity())return;const button=form.querySelector('button[type="submit"]');if(button){button.textContent=button.dataset.loadingLabel || (button.textContent.includes('Ask')?'Thinking…':'Saving…');button.disabled=true;}});
const checkin=document.querySelector('#check-in-section form');
if(checkin){const fields=[...checkin.querySelectorAll('input:not([type="hidden"]):not([type="checkbox"]),select')];const update=()=>{const count=fields.filter(field=>field.value.trim()!=='').length;const value=Math.round(count/Math.max(1,fields.length)*100);const bar=document.querySelector('.checkin-progress span');if(bar)bar.style.width=value+'%';const label=document.getElementById('checkin-completion');if(label)label.textContent=value+'% filled in';};checkin.addEventListener('input',update);update();}
for(const meter of document.querySelectorAll('[data-risk-percent]'))meter.style.width=Math.max(0,Math.min(100,Number(meter.dataset.riskPercent)))+'%';
for(const button of document.querySelectorAll('[data-condition-filter]'))button.addEventListener('click',()=>{
    const selected=button.dataset.conditionFilter;
    for(const card of document.querySelectorAll('[data-condition]'))card.hidden=selected!=='all' && card.dataset.condition!==selected;
    for(const option of document.querySelectorAll('[data-condition-filter]'))option.setAttribute('aria-pressed',String(option===button));
});
for (const area of document.querySelectorAll('textarea')) {
    const fit = () => {
        area.style.height = 'auto';
        area.style.height = Math.max(44, area.scrollHeight + 2) + 'px';
    };
    area.addEventListener('input', fit);
    window.addEventListener('resize', fit);
    fit();
}
