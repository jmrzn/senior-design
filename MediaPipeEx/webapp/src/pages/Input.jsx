
import { useState } from 'react'
import '../index.css'

function Input({ dance, onSelectMode, onBack }) {
  const [inputMode, setInputMode] = useState(null);

  return (
    <div className='app-container'>
      <button onClick={onBack}>← Back</button>
      <h2>Practicing: {dance}</h2>
      
      <div className="options">
        {/* Record from your camera: */}
        <button onClick={() => onSelectMode('record')}>Record</button>
        {/* Upload a file from your computer: */}
        <button onClick={() => onSelectMode('upload')}>Upload</button>
      </div>

      {inputMode && <div>You chose to {inputMode}</div>}
    </div>
  );
}

export default Input