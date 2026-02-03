import { useEffect, useState } from 'react'
import '../index.css'

function Feedback({ videoFile, danceName, onRestart }) {
  const [processedUrl, setProcessedUrl] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const uploadVideo = async () => {
      const formData = new FormData();
      formData.append('video', videoFile);

      try {
        const response = await fetch('http://localhost:5000/process-video', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Network response was not ok');
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setProcessedUrl(url);
        setLoading(false);
      } catch (error) {
        console.error("Error processing video:", error);
        setLoading(false);
      }
    };

    if (videoFile) uploadVideo();
  }, [videoFile]);

  return (
    <div className="feedback-page">
      {loading ? (
        <div className="loader">
          <h2>Analyzing your moves...</h2>
          <div className="progress-bar-container">
             <div className="progress-bar-fill"></div>
          </div>
        </div>
      ) : (
        <div className="feedback-container">
          <h2>Feedback: {danceName}</h2>
          <video src={processedUrl} controls autoPlay  />
          
          <button onClick={onRestart}>Start New Dance</button>
        </div>
      )}
    </div>
  );
}

export default Feedback;