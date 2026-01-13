import { useState } from 'react'

function Start() {
  const [dance, setDance] = useState('Dance 1')

  return (
    <>
      <div className='Title'> 
        Dance!
      </div>
      <div className='caption'> 
        Compare your dances to your instructors and get curated and specific feedback on how you can improve.
      </div>  
      <div className='callToAction'> 
        Choose which dance you are practicing!
      </div>    
        <button onClick={() => setDance('Dance 1')}>
            Dance 1
        </button>
        <button onClick={() => setDance('Dance 2')}>
            Dance 2
        </button>
        <button onClick={() => setDance('Dance 3')}>
            Dance 3
        </button>
        <div>You chose {dance}</div>
        
    </>
  )
}

export default Start
