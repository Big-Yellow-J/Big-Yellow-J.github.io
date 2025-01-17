---
layout: mypost
title: 关于
---
# Who

Hi！欢迎来自<span id="visitor-location">某地</span>

我是黄杰  

我现在在（不妨等一等谷歌加载哈哈哈）：  

<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d13187.62315506682!2d114.3654708839818!3d30.47356738111945!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x342ebb0327eda313%3A0x4ca810852fdd8295!2z5Lit5Y2X6LSi57uP5pS_5rOV5aSn5a2m5Y2X5rmW5qCh5Yy656CU56m255Sf6Zmi!5e0!3m2!1szh-CN!2sjp!4v1737095885217!5m2!1szh-CN!2sjp" width="400" height="300" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>

读研究生二年级！  

主要研究兴趣是：**文档AI**。研究生期间没有发表过 *KDD*，也没发表过 *NIPS*，更加没有发表过 *CVPR*😄😄😄😄😄  
**但是**：  
发表过若干Blog😄😄😄😄😄😄😄  

# 联系我  

- Email1&nbsp;: [hjie20011001@gmail.com](mailto:hjie20011001@gmail.com)  
- Email2&nbsp;: [2802311325@qq.com](mailto:2802311325@gmail.com)  
- GitHub: [https://github.com/shangxiaaabb](https://github.com/shangxiaaabb) 


平时喜欢做韭菜（纯被割韭！！！）所以让我们关注一下今天韭菜是涨还是跌！

<div class="stock-container">
  <div class="stock-name" id="stock-name">上证指数</div>
  <div class="stock-price" id="stock-price">加载中...</div>
  <div class="stock-change" id="stock-change"></div>
</div>

<style>
  .stock-container {
    background-color: #fff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    text-align: center;
    margin-top: 20px;
  }
  .stock-name {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 10px;
  }
  .stock-price {
    font-size: 32px;
    font-weight: bold;
  }
  .stock-change {
    font-size: 18px;
    margin-top: 10px;
  }
  .up {
    color: #ff4d4d; /* 上涨为红色 */
  }
  .down {
    color: #00cc66; /* 下跌为绿色 */
  }
</style>

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

  function fetchStockData() {
  const stockNameElement = document.getElementById('stock-name');
  const stockPriceElement = document.getElementById('stock-price');
  const stockChangeElement = document.getElementById('stock-change');

  // 腾讯财经 API URL（上证指数代码：sh000001）
  const tencentUrl = 'http://qt.gtimg.cn/q=sh000001';

  // 新浪财经 API URL（上证指数代码：s_sh000001）
  const sinaUrl = 'https://hq.sinajs.cn/list=s_sh000001';

  // 雪球 API URL（上证指数代码：SH000001）
  const xueqiuUrl = 'https://stock.xueqiu.com/v5/stock/quote.json?symbol=SH000001&extend=detail';

  // 创建 AbortController 用于超时控制
  const controller = new AbortController();
  const signal = controller.signal;

  // 设置超时时间（3 秒）
  const timeout = 3000;

  // 超时处理
  const timeoutId = setTimeout(() => {
    controller.abort(); // 中止腾讯财经 API 请求
    console.warn('腾讯财经 API 请求超时，切换到新浪财经 API'); // 调试日志
    fetchStockDataFromSina(); // 切换到新浪财经 API
  }, timeout);

  // 尝试使用腾讯财经 API
  fetch(tencentUrl, { signal })
    .then(response => response.text())
    .then(data => {
      clearTimeout(timeoutId); // 清除超时计时器
      console.log('腾讯财经 API 返回数据:', data); // 调试日志
      // 解析返回的数据（格式为 CSV）
      const parts = data.split('~');
      if (parts.length > 1) {
        const indexName = parts[1]; // 指数名称
        const currentPrice = parts[3]; // 当前价格
        const change = parts[4]; // 涨跌额
        const changePercent = parts[5]; // 涨跌百分比

        // 设置颜色样式
        const isUp = parseFloat(change) > 0;
        stockPriceElement.className = isUp ? 'stock-price up' : 'stock-price down';
        stockChangeElement.className = isUp ? 'stock-change up' : 'stock-change down';

        // 显示数据
        stockNameElement.innerText = indexName;
        stockPriceElement.innerText = currentPrice;
        stockChangeElement.innerText = `${change} (${changePercent})`;
      } else {
        throw new Error('腾讯财经 API 数据解析失败');
      }
    })
    .catch(error => {
      if (error.name === 'AbortError') {
        console.warn('腾讯财经 API 请求被中止，已切换到新浪财经 API'); // 调试日志
      } else {
        console.error('腾讯财经 API 请求失败:', error); // 调试日志
      }
      fetchStockDataFromSina(); // 切换到新浪财经 API
    });

  // 使用新浪财经 API 获取数据
  function fetchStockDataFromSina() {
    fetch(sinaUrl)
      .then(response => response.text())
      .then(data => {
        console.log('新浪财经 API 返回数据:', data); // 调试日志
        // 解析返回的数据（格式为 CSV）
        const parts = data.split(',');
        if (parts.length > 1) {
          const indexName = parts[0].split('"')[1]; // 指数名称
          const currentPrice = parts[1]; // 当前价格
          const change = parts[2]; // 涨跌额
          const changePercent = parts[3]; // 涨跌百分比

          // 设置颜色样式
          const isUp = parseFloat(change) > 0;
          stockPriceElement.className = isUp ? 'stock-price up' : 'stock-price down';
          stockChangeElement.className = isUp ? 'stock-change up' : 'stock-change down';

          // 显示数据
          stockNameElement.innerText = indexName;
          stockPriceElement.innerText = currentPrice;
          stockChangeElement.innerText = `${change} (${changePercent})`;
        } else {
          throw new Error('新浪财经 API 数据解析失败');
        }
      })
      .catch(error => {
        console.error('新浪财经 API 请求失败:', error); // 调试日志
        fetchStockDataFromXueqiu(); // 切换到雪球 API
      });
  }

  // 使用雪球 API 获取数据
  function fetchStockDataFromXueqiu() {
    fetch(xueqiuUrl)
      .then(response => response.json())
      .then(data => {
        console.log('雪球 API 返回数据:', data); // 调试日志
        if (data.data && data.data.quote) {
          const quote = data.data.quote;
          const indexName = quote.name; // 指数名称
          const currentPrice = quote.current; // 当前价格
          const change = quote.chg; // 涨跌额
          const changePercent = quote.percent; // 涨跌百分比

          // 设置颜色样式
          const isUp = parseFloat(change) > 0;
          stockPriceElement.className = isUp ? 'stock-price up' : 'stock-price down';
          stockChangeElement.className = isUp ? 'stock-change up' : 'stock-change down';

          // 显示数据
          stockNameElement.innerText = indexName;
          stockPriceElement.innerText = currentPrice;
          stockChangeElement.innerText = `${change} (${changePercent}%)`;
        } else {
          throw new Error('雪球 API 数据解析失败');
        }
      })
      .catch(error => {
        console.error('雪球 API 请求失败:', error); // 调试日志
        stockPriceElement.innerText = '数据加载失败';
      });
  }
}

  // 页面加载时执行
  window.onload = function() {
    getLocation(); // 获取访问者地理位置
    fetchStockData(); // 获取上证指数数据
  };
</script>