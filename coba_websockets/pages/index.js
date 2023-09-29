// pages/index.js
'use client'
import WebcamComponent from '../components/WebcamComponent';
import { useRouter } from 'next/navigation';

const HomePage = () => {
const router = useRouter()
 
  
  return (
    <div>
      <h1>Webcam Example</h1>
      <WebcamComponent />
      <button onClick={()=> router.push('/test')}>Test</button>
    </div>
  );
}; 

export default HomePage;
