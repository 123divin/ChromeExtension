chrome.runtime.onInstalled.addListener(() => {
    console.log('Phishing Detection Extension Installed');
  });
  
  chrome.action.onClicked.addListener((tab) => {
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      function: detectPhishing
    });
  });
  
  async function detectPhishing() {
    const url = window.location.href;
  
    // Send URL to backend for prediction
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: url, html_content: document.documentElement.outerHTML })
    });
  
    const result = await response.json();
  
    if (result.is_phishing) {
      alert('Warning: This site might be a phishing site!');
    } else {
      alert('This site seems safe.');
    }
  }
  