import React, { useState } from 'react';
import './App.css'; // You can use this for basic styling

// --- CRITICAL: REPLACE THIS PLACEHOLDER WITH YOUR ACTUAL API ENDPOINT ---
const API_ENDPOINT = "/2015-03-31/functions/function/invocations";  

function NewsValidator() {
  // State for the text input by the user
  const [newsContent, setNewsContent] = useState('');
  
  // State for the prediction result
  const [prediction, setPrediction] = useState(null);
  
  // State to track loading status
  const [isLoading, setIsLoading] = useState(false);
  
  // State for any API errors
  const [error, setError] = useState(null);

  /**
   * Handles the click event for the Verify button.
   */
  const handleVerify = async () => {
    if (!newsContent.trim()) {
      alert("Please enter news content to verify.");
      return;
    }

    setIsLoading(true);
    setPrediction(null);
    setError(null);
    
    // The JSON body structure expected by your Python Lambda handler.py
    const requestBody = {
      text: newsContent
    };

    try {
      // 1. Make the POST request to the AWS API Gateway endpoint
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Note: API Gateway must have CORS enabled for this to work from S3!
        },
        body: JSON.stringify(requestBody),
      });

      // Handle HTTP errors (e.g., 4xx, 5xx)
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      // 2. Parse the JSON response from the Lambda function
      const data = await response.json();
      
      // The Lambda response body is returned as a JSON string, so we need to parse it again
      // if Lambda uses the standard proxy integration and strings the 'body' field.
      // (Assuming your Lambda returns a parsed JSON object directly as in handler.py example)
      const result = typeof data.body === 'string' ? JSON.parse(data.body) : data;
      
      setPrediction(result);
      
    } catch (err) {
      console.error("Verification failed:", err);
      setError("Failed to connect to the model API. Check CORS or endpoint URL.");
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Renders the classification result based on the prediction state.
   */
  const renderResult = () => {
    if (error) {
      return <div className="result-error">Error: {error}</div>;
    }
    
    if (prediction) {
      const { is_fake, probabilities } = prediction;
      const resultText = is_fake ? 'FAKE NEWS' : 'REAL NEWS';
      const resultClass = is_fake ? 'fake' : 'real';

      return (
        <div className={`prediction-result ${resultClass}`}>
          <h2>Classification Result: <span className={resultClass}>{resultText}</span></h2>
          <p>Confidence: **Fake:** {(probabilities.fake * 100).toFixed(2)}% | **Real:** {(probabilities.real * 100).toFixed(2)}%</p>
          <p className="note">Interpretation: The model uses TF-IDF features to determine classification. (See Project Report for XAI details)</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="container">
      <h1>Fake News Detection Validator</h1>
      <p>Enter a news article or snippet below to verify its authenticity.</p>

      <div className="input-section">
        <textarea
          rows="10"
          placeholder="Paste news content here..."
          value={newsContent}
          onChange={(e) => setNewsContent(e.target.value)}
          disabled={isLoading}
        />
        
        <button 
          onClick={handleVerify} 
          disabled={isLoading}
        >
          {isLoading ? 'Verifying...' : 'Validate News'}
        </button>
      </div>

      <div className="output-section">
        {isLoading && <div className="loading">Analyzing content...</div>}
        {renderResult()}
      </div>
    </div>
  );
}



// To use this code, you can either:
// 1. Export it directly as App (if placed in App.js)
// 2. Put the CSS into your App.css file.
// 
// For a quick test:
// function App() { return ( <NewsValidator /> ); }
// export default App; 

export default NewsValidator;