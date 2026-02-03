import { useState } from 'react'
import '../index.css'

function Upload({ onBack, onVideoReady }) {
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type.startsWith('video/')) {
      setFile(selectedFile);
    } else {
      alert("Please upload a valid video file.");
    }
  };

  const handleUpload = () => {
    if (file) {
      onVideoReady(file); // Send the file to the parent for processing
    }
  };

  return (
    <div className="app-container">
      <button onClick={onBack}>← Back</button>
      <h2>Upload Dance Video</h2>
      
      <div className="upload-box" style={{ border: '2px dashed #ccc', padding: '40px', margin: '20px 0' }}>
        <input type="file" accept="video/*" onChange={handleFileChange} />
        {file && <p>Selected: {file.name}</p>}
      </div>

      <button onClick={handleUpload} disabled={!file}>
        Proceed to Feedback
      </button>
    </div>
  );
}

export default Upload;