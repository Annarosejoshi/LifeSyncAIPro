const $ = id => document.getElementById(id);
if (typeof Chart !== "undefined") {
    Chart.defaults.font.size = 16;
    Chart.defaults.font.family = 'Arial, Helvetica, sans-serif';
}
const dashboardData = JSON.parse($("dashboard-data").textContent);
function text(id,value){if($(id))$(id).textContent=value ?? "—";}
text("today",new Date().toLocaleDateString("en-IN",{timeZone:"Asia/Kolkata",day:"numeric",month:"short",year:"numeric"}));
for(const [id,key] of [["sleepVal","sleep"],["waterVal","water"],["exerciseVal","exercise"],["moodVal","mood"],["stressVal","stress"],["nutritionVal","nutrition"]])text(id,dashboardData[key]);
text("mealsVal",dashboardData.mealsCompleted == null ? "— / 3" : dashboardData.mealsCompleted+" / 3");
const bmi = dashboardData.height > 0 && dashboardData.weight > 0 ? dashboardData.weight / (dashboardData.height/100)**2 : null;
text("bmiVal",bmi == null ? null : bmi.toFixed(2));
if(bmi != null)text("bmiStatus","From saved profile measurements");
for(const [id,value,target] of [["sleepBar",dashboardData.sleep,8],["waterBar",dashboardData.water,2.5],["exerciseBar",dashboardData.exercise,30],["mealsBar",dashboardData.mealsCompleted,3],["stressBar",dashboardData.stress,5],["nutritionBar",dashboardData.nutrition,5]])if($(id))$(id).style.width=Math.min(100,Math.max(0,(value ?? 0)/target*100))+"%";

$("printSummary")?.addEventListener("click",()=>window.print());
for(const link of document.querySelectorAll(".nav"))if(new URL(link.href).pathname===location.pathname)link.classList.add("active");
function createChart(canvasId){
  const canvas=$(canvasId);
  if(!canvas||typeof Chart==="undefined")return;
  new Chart(canvas,{
    type:"line",
    data:{
      labels:dashboardData.chartLabels,
      datasets:[
        {label:"Sleep target (%)",data:dashboardData.sleepTrend,borderColor:"#8b72e9",backgroundColor:"#8b72e922",tension:.35},
        {label:"Hydration target (%)",data:dashboardData.waterTrend,borderColor:"#42a5f5",backgroundColor:"#42a5f522",tension:.35},
        {label:"Exercise target (%)",data:dashboardData.exerciseTrend,borderColor:"#24b78a",backgroundColor:"#24b78a22",tension:.35},
        {label:"Health score",data:dashboardData.scoreTrend,borderColor:"#e19443",backgroundColor:"#e1944322",tension:.35}
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:"bottom",labels:{usePointStyle:true,boxWidth:7,font:{size:16}}}},
      scales:{
        y:{min:0,max:100,ticks:{stepSize:25,font:{size:16}},grid:{color:"#edf1ed"}},
        x:{grid:{display:false},ticks:{font:{size:16}}}
      }
    }
  });
}

function createBmiChart(){
  const canvas=$("bmiHistoryChart");
  if(!canvas||typeof Chart==="undefined"||!dashboardData.bmiChartValues.length)return;
  new Chart(canvas,{
    type:"line",
    data:{
      labels:dashboardData.bmiChartLabels,
      datasets:[{
        label:"BMI history",
        data:dashboardData.bmiChartValues,
        borderColor:"#087f68",
        backgroundColor:"#087f6822",
        pointBackgroundColor:"#087f68",
        tension:.3,
        fill:true
      }]
    },
    options:{
      responsive:true,
      maintainAspectRatio:false,
      plugins:{legend:{position:"bottom"}},
      scales:{x:{grid:{display:false}},y:{beginAtZero:false}}
    }
  });
}

const trendCanvas = $("trendChart");
if(trendCanvas && typeof Chart !== "undefined") {
    const choices = {sleep:["Sleep (hours)",[dashboardData.sleep],"#8b72e9"],water:["Water (litres)",[dashboardData.water],"#42a5f5"],exercise:["Exercise (minutes)",[dashboardData.exercise],"#24b78a"],score:["Health score",[dashboardData.healthScore],"#e19443"]};
    const chart = new Chart(trendCanvas,{type:"bar",data:{labels:["Today"],datasets:[]},options:{responsive:true,maintainAspectRatio:false,scales:{y:{beginAtZero:true}},plugins:{legend:{display:false}}}});
    function updateTrend(){const [label,data,color]=choices[$("trendMetric").value];chart.data.datasets=[{label,data,borderColor:color,backgroundColor:color+"88",borderWidth:1,borderRadius:10,maxBarThickness:100}];chart.options.scales.y.title={display:true,text:label};text("dailyChartValue",label+": "+data[0]);chart.update();}
    $("trendMetric").addEventListener("change",updateTrend);updateTrend();
}
createBmiChart();
