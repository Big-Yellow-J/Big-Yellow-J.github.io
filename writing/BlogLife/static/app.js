// 纯前端增强：头像配色、相对时间、上传预览、图片放大。零依赖。

// 头像按名字生成稳定色相，避免大家都一个色
function hue(s){let h=0;for(const c of s)h=(h*31+c.charCodeAt(0))%360;return h}
document.querySelectorAll('.avatar[data-name]').forEach(el=>{
  el.style.background = `hsl(${hue(el.dataset.name)},52%,55%)`;
});

// 相对时间："刚刚 / x分钟前 / x小时前 / 昨天 / 日期"
function rel(ts){
  const d = Date.now()/1000 - ts;
  if(d < 60)    return '刚刚';
  if(d < 3600)  return Math.floor(d/60)  + '分钟前';
  if(d < 86400) return Math.floor(d/3600)+ '小时前';
  if(d < 172800)return '昨天';
  const t = new Date(ts*1000);
  return `${t.getMonth()+1}月${t.getDate()}日`;
}
document.querySelectorAll('.time[data-ts]').forEach(el=>{
  const ts = +el.dataset.ts;
  el.textContent = rel(ts);
  el.title = new Date(ts*1000).toLocaleString();   // 悬停看精确时间
});

// 选图后本地预览 + 可移除
const fileInput = document.querySelector('input[type=file]');
const preview   = document.getElementById('preview');
if(fileInput && preview){
  const pimg = preview.querySelector('img');
  fileInput.addEventListener('change', ()=>{
    const f = fileInput.files[0];
    if(f && f.type.startsWith('image/')){           // HEIC 浏览器可能不预览，仅 type 命中才显示
      pimg.src = URL.createObjectURL(f);
      preview.hidden = false;
    }else{ preview.hidden = true; }
  });
  preview.querySelector('.rm').addEventListener('click', ()=>{
    fileInput.value = ''; preview.hidden = true;
  });
}

// 点图放大，点任意处关闭
const lb = document.getElementById('lightbox');
if(lb){
  const lbimg = lb.querySelector('img');
  document.querySelectorAll('.photo').forEach(img=>{
    img.addEventListener('click', ()=>{ lbimg.src = img.src; lb.hidden = false; });
  });
  lb.addEventListener('click', ()=>{ lb.hidden = true; lbimg.src=''; });
}
