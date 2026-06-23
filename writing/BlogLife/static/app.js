// 纯前端增强：头像配色、相对时间、多图预览、点赞(异步)、回复、删除确认、图片放大左右切换。零依赖。

// 头像按名字生成稳定色相
function hue(s){let h=0;for(const c of s)h=(h*31+c.charCodeAt(0))%360;return h}
document.querySelectorAll(".avatar[data-name]").forEach(function(el){
  el.style.background = "linear-gradient(135deg,hsl("+hue(el.dataset.name)+",70%,62%),hsl("+((hue(el.dataset.name)+40)%360)+",70%,55%))";
});

// 相对时间
function rel(ts){
  var d = Date.now()/1000 - ts;
  if(d < 60)    return "刚刚";
  if(d < 3600)  return Math.floor(d/60)  + "分钟前";
  if(d < 86400) return Math.floor(d/3600)+ "小时前";
  if(d < 172800)return "昨天";
  var t = new Date(ts*1000);
  return (t.getMonth()+1) + "月" + t.getDate() + "日";
}
document.querySelectorAll(".time[data-ts]").forEach(function(el){
  var ts = +el.dataset.ts;
  el.textContent = rel(ts);
  el.title = new Date(ts*1000).toLocaleString();
});

// 选图后多图预览
var fileInput = document.querySelector(".composer input[type=file]");
var preview   = document.getElementById("preview");
if(fileInput && preview){
  var picked = [];                              // 累积多次选择，避免原生 input 覆盖
  function sync(){
    var dt = new DataTransfer();                // 写回 input.files，提交时全部上传
    picked.forEach(function(f){ dt.items.add(f); });
    fileInput.files = dt.files;
    preview.innerHTML = "";
    picked.forEach(function(f, i){
      var box = document.createElement("div"); box.className = "pv";
      var img = document.createElement("img"); img.src = URL.createObjectURL(f);
      var rm  = document.createElement("span"); rm.className = "pv-rm"; rm.textContent = "\u00d7";
      rm.addEventListener("click", function(){ picked.splice(i, 1); sync(); });   // 单张删除
      box.appendChild(img); box.appendChild(rm); preview.appendChild(box);
    });
  }
  fileInput.addEventListener("change", function(){
    Array.prototype.forEach.call(fileInput.files, function(f){
      if(f.type.startsWith("image/") || /\.(heic|heif)$/i.test(f.name)) picked.push(f);
    });
    sync();
  });
}

// 点赞：异步切换红心 + 数字，不刷新
document.querySelectorAll(".like-btn").forEach(function(btn){
  btn.addEventListener("click", function(){
    fetch("/like", {method:"POST",
      headers:{"Content-Type":"application/x-www-form-urlencoded"},
      body:"pid=" + btn.dataset.pid})
      .then(function(r){return r.json();})
      .then(function(j){
        btn.classList.toggle("liked", j.liked);
        btn.querySelector(".heart").textContent = j.liked ? "\u2665" : "\u2661";
        btn.querySelector(".like-count").textContent = j.count;
      });
  });
});

// 回复评论：预填评论框 + 设 reply_to
document.querySelectorAll(".reply-btn").forEach(function(btn){
  btn.addEventListener("click", function(){
    var form = btn.closest(".card").querySelector(".cmt-form");
    form.reply_to.value = btn.dataset.nick;
    var inp = form.querySelector(".cmt-input");
    inp.placeholder = "回复 @" + btn.dataset.nick + "…";
    inp.focus();
  });
});

// 删除动态二次确认
document.querySelectorAll(".del-form").forEach(function(f){
  f.addEventListener("submit", function(e){ if(!confirm("删除这条动态？不可恢复")) e.preventDefault(); });
});

// 图片放大 + 左右切换（同一条动态的多图为一组）
var lb     = document.getElementById("lightbox");
var lbImg  = lb.querySelector(".lb-img");
var lbCnt  = lb.querySelector(".lb-count");
var lbPrev = lb.querySelector(".lb-prev");
var lbNext = lb.querySelector(".lb-next");
var group = [], cur = 0;
function lbShow(){
  lbImg.src = group[cur];
  lbCnt.textContent = (cur+1) + "/" + group.length;
  var multi = group.length > 1 ? "flex" : "none";
  lbPrev.style.display = multi; lbNext.style.display = multi;
  lbCnt.style.display  = group.length > 1 ? "block" : "none";
}
function lbOpen(list, i){ group = list; cur = i; lb.hidden = false; lbShow(); }
// 事件委托：点任意 .photo 都能放大，不依赖绑定时机
document.addEventListener("click", function(e){
  var im = e.target.closest ? e.target.closest(".photo") : null;
  if(!im) return;
  var imgs = Array.prototype.slice.call(im.closest(".photos").querySelectorAll(".photo"));
  lbOpen(imgs.map(function(x){ return x.src; }), imgs.indexOf(im));
});
lbPrev.addEventListener("click", function(e){ e.stopPropagation(); cur=(cur-1+group.length)%group.length; lbShow(); });
lbNext.addEventListener("click", function(e){ e.stopPropagation(); cur=(cur+1)%group.length; lbShow(); });
lb.addEventListener("click", function(e){ if(e.target === lb) lb.hidden = true; });   // 点背景关
document.addEventListener("keydown", function(e){
  if(lb.hidden) return;
  if(e.key === "Escape")     lb.hidden = true;
  if(e.key === "ArrowLeft")  lbPrev.click();
  if(e.key === "ArrowRight") lbNext.click();
});


// 移动端：lightbox 左右滑动切换
var _tx = 0;
lb.addEventListener("touchstart", function(e){ _tx = e.changedTouches[0].clientX; }, {passive:true});
lb.addEventListener("touchend", function(e){
  var dx = e.changedTouches[0].clientX - _tx;
  if(Math.abs(dx) > 40){ if(dx < 0) lbNext.click(); else lbPrev.click(); }
}, {passive:true});
