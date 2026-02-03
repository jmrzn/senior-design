import { useState, useRef } from 'react'
import '../index.css'

function Record({ onBack, onVideoReady }) {
  const [recording, setRecording] = useState(false);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recordedChunks, setRecordedChunks] = useState([]);

  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
  };

  const startRecording = () => {
    setRecording(true);
    setRecordedChunks([]);
    const stream = videoRef.current.srcObject;
    mediaRecorderRef.current = new MediaRecorder(stream);
    
    mediaRecorderRef.current.ondataavailable = (e) => {
      if (e.data.size > 0) setRecordedChunks((prev) => [...prev, e.data]);
    };
    
    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      onVideoReady(blob); // Pass the video file up to be processed
    };

    mediaRecorderRef.current.start();
  };

  const stopRecording = () => {
    setRecording(false);
    mediaRecorderRef.current.stop();
  };

  return (
    <div className="record-container">
      <button onClick={onBack}>← Back</button>
      <h2>Record Your Dance</h2>
      
      <div className="video-preview" style={{ background: '#000', width: '100%', maxWidth: '500px', height: '300px' }}>
        <video ref={videoRef} autoPlay muted playsInline style={{ width: '100%' }} />
      </div>

      <div className="controls">
        {!videoRef.current?.srcObject && <button onClick={startCamera}>Open Camera</button>}
        
        {videoRef.current?.srcObject && !recording && (
          <button onClick={startRecording} style={{ background: 'red', color: 'white' }}>Start Recording</button>
        )}
        
        {recording && (
          <button onClick={stopRecording}>Stop & Save</button>
        )}
      </div>
    </div>
  );
}

export default Record;