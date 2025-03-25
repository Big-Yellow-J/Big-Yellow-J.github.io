---
layout: mypost
title: 关于
show_footer_image: false
---
# Who

Hi！欢迎来自<span id="visitor-location">某地</span>

我是黄杰  

我现在在：  

<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d13187.62315506682!2d114.3654708839818!3d30.47356738111945!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x342ebb0327eda313%3A0x4ca810852fdd8295!2z5Lit5Y2X6LSi57uP5pS_5rOV5aSn5a2m5Y2X5rmW5qCh5Yy656CU56m255Sf6Zmi!5e0!3m2!1szh-CN!2sjp!4v1737095885217!5m2!1szh-CN!2sjp" width="400" height="300" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>


读研究生二年级！  

主要研究兴趣是：**文档AI**，**文档智能解析**。研究生期间没有发表过 *KDD*，也没发表过 *NIPS*，更加没有发表过 *CVPR*😄😄😄😄😄  
**但是**：  
发表过若干Blog😄😄😄😄😄😄😄 
我的技能以及常用的工具

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,vscode,git,linux,pytorch,raspberrypi,ubuntu,docker,sklearn&theme=light" alt="Skill Icons">
</p>


# 联系我  

- Email1&nbsp;: [hjie20011001@gmail.com](mailto:hjie20011001@gmail.com)  
- Email2&nbsp;: [2802311325@qq.com](mailto:2802311325@gmail.com)  
- GitHub: [https://github.com/shangxiaaabb](https://github.com/shangxiaaabb) 


 <script>
  // 获取访问者地理位置
  function fetchAddress(lat, lon) {
    const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}&accept-language=en`;
    fetch(url)
      .then(response => response.json())
      .then(data => {
        const location =
          data.address.city ||
          data.address.town ||
          data.address.village ||
          "某地";
        document.getElementById("visitor-location").textContent = location;
      })
      .catch(() => {
        document.getElementById("visitor-location").textContent = "某地";
      });
  }
  function getLocation() {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const lat = position.coords.latitude;
          const lon = position.coords.longitude;
          fetchAddress(lat, lon);
        },
        () => {
          document.getElementById("visitor-location").textContent = "某地";
        }
      );
    } else {
      document.getElementById("visitor-location").textContent = "某地";
    }
  }

  // 页面加载时执行
  window.onload = function() {
    getLocation(); // 获取访问者地理位置
  };
</script>