import '../index.css'

function Start({ onSelectDance }) {
  return (
    <div className='app-container'>
      <div className='Title'>Dance!</div>
      <div className='Caption'> Compare your dances to your instructors and get curated and specific feedback on how you can improve.</div>  
      <div className='CallToAction'>Choose which dance you are practicing!</div>

      
      {/* Call the parent function when clicked */}
      <button onClick={() => onSelectDance('Dance 1')}>Dance 1</button>
      <button onClick={() => onSelectDance('Dance 2')}>Dance 2</button>
      <button onClick={() => onSelectDance('Dance 3')}>Dance 3</button>
    </div>
  );
}

export default Start