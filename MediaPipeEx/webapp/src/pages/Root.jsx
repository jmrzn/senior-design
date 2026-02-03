import { useState } from 'react';
import Start from './Start';
import Input from './Input';
import Record from './Record';
import Upload from './Upload';
import Feedback from './Feedback';

function Root() {
  const [page, setPage] = useState('start'); 
  const [selectedDance, setSelectedDance] = useState(null);
  const [userVideo, setUserVideo] = useState(null);

  const goToInput = (dance) => {
    setSelectedDance(dance);
    setPage('input');
  }

  const handleVideo = (video) => {
    setUserVideo(video);
    setPage('feedback');
  }
  // const handleDanceSelect = (danceName) => {
  //   setSelectedDance(danceName);
  //   setPage('feedback'); 
  // };

  const onSelectMode = (mode) => {
    setPage(mode)
  }

  return (
    <div className='app-container'>
      {page === 'start' && (
        <Start onSelectDance={goToInput} />
      )}

      {page === 'input' && (
        <Input 
          dance={selectedDance}
          onSelectMode={onSelectMode}
          onBack={() => setPage('start')} 
        />
      )}

      {page === 'record' && (
        <Record 
          onBack={() => setPage('input')} 
          onVideoReady={handleVideo}/>
      )}

      {page === 'upload' && (
        <Upload 
          onBack={() => setPage('input')} 
          onVideoReady={handleVideo}/>
      )}

      {page === 'feedback' && (
        <Feedback 
          videoFile={userVideo}
          danceName={selectedDance}
          onRestart={() => setPage('start')}/>
      )}
    </div>
  );
}

export default Root;