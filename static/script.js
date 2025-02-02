document.getElementById('linkForm').addEventListener('submit', function(event) {
  event.preventDefault();
  const reelLink = document.getElementById('reelLink').value;

  // Display loading message
  document.getElementById('output').innerText = "Processing...";

  // Call backend API
  fetch('/extract-info', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({ link: reelLink })
  })
  .then(response => response.json())
  .then(data => {
      if (data.info) {
          document.getElementById('output').innerText = data.info;
      } else {
          document.getElementById('output').innerText = "Error extracting information.";
      }
  })
  .catch(error => {
      document.getElementById('output').innerText = "An error occurred.";
      console.error(error);
  });
});
