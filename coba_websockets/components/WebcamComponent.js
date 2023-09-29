// components/WebcamComponent.js
import { useEffect, useRef, useState } from 'react';

const WebcamComponent = () => {
  const webcamRef = useRef(null);
  const serverUrl = 'ws://localhost:8765';  // Update with your server URL
  const [serverResponse, setServerResponse] = useState('');

  useEffect(() => {
    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (webcamRef.current) {
          webcamRef.current.srcObject = stream;

          // Capture and send image every second
          setInterval(() => {
            captureAndSendImage();
          }, 1000);
        }
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    };

    const captureAndSendImage = async () => {
      const image = captureImage();
      sendImageToServer(image);
    };

    const captureImage = () => {
      const canvas = document.createElement('canvas');
      canvas.width = webcamRef.current.videoWidth;
      canvas.height = webcamRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      
      // Flip the image horizontally
      ctx.scale(-1, 1);
      ctx.drawImage(webcamRef.current, 0, 0, -canvas.width, canvas.height);
      
      // Reset the transformation to avoid affecting other drawings
      ctx.setTransform(1, 0, 0, 1, 0, 0);

      return canvas.toDataURL('image/jpeg');
    };

    const sendImageToServer = (image) => {
      const socket = new WebSocket(serverUrl);

      socket.onopen = () => {
        console.log('WebSocket connection opened');
        socket.send(image);
      };

      socket.onmessage = (event) => {
        console.log('Received from server:', event.data);
        // Handle the server response
        setServerResponse(event.data);
      };

      socket.onclose = (event) => {
        console.log('WebSocket connection closed:', event);
      };
    };

    startWebcam();

    return () => {
      // Stop the webcam when the component is unmounted
      const tracks = webcamRef.current?.srcObject?.getTracks();
      tracks && tracks.forEach(track => track.stop());
    };
  }, []);

  return (
    <div>
      <video ref={webcamRef} autoPlay playsInline style={{ transform: 'scaleX(-1)' }} />
      <p>Server Response: {serverResponse}</p>
    </div>
  );
};

export default WebcamComponent;