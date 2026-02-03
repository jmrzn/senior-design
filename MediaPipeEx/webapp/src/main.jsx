import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'

import Root from './pages/Root'
import Record from './pages/Record'
import Upload from './pages/Upload'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)
