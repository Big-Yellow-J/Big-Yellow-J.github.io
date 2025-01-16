---
layout: mypost
title: 关于
---
# Who
Hi！欢迎来自<span id="visitor-location">某地</span>的朋友！我是黄杰，研究生二年级在读！  
主要研究兴趣是：**文档AI**。研究生期间没有发表过 *KDD*，也没发表过 *NIPS*，更加没有发表过 *CVPR*😄😄😄😄😄  
**但是**：  
发表过若干Blog😄😄😄😄😄😄😄  
我现在在（不妨等一等谷歌加载哈哈哈）：  

<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d439724.63177137234!2d113.97072902668832!3d30.567700731809726!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x342ebb1084f8e049%3A0xa644e7861424aee3!2sZhongnan%20University%20of%20Economics%20and%20Law!5e0!3m2!1sen!2sjp!4v1737037826235!5m2!1sen!2sjp" width="400" height="300" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>  

读研究生二年级！  

# 联系我  

- Email&nbsp;: [hjie20011001@gmail.com](mailto:hjie20011001@gmail.com)  

- GitHub: [https://github.com/shangxiaaabb](https://github.com/shangxiaaabb)  

<script>
  function fetchAddress(lat, lon) {
    const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}&accept-language=zh-CN`;
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

  window.onload = getLocation;
</script>
