import React, { useState } from 'react';
import axios from 'axios';
import { Container, Form, Button, Image } from 'react-bootstrap';
import './styles.css'; // Import the custom styles

const App = () => {
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null); // State to manage video preview
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false); // State to manage loading state

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setVideoFile(file);
    setVideoPreview(URL.createObjectURL(file)); // Set video preview URL
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    const formData = new FormData();
    formData.append('video', videoFile);

    try {
      setLoading(true); // Start loading state

      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data.prediction);
      setConfidence(response.data.confidence);
      setImage(`data:image/png;base64,${response.data.image}`);
    } catch (error) {
      console.error('Error uploading video:', error);
    } finally {
      setLoading(false); // Stop loading state
    }
  };

  return (
    <>
      <div className="outside-container">
        <h1>arma AI</h1>
        {/* <p>Upload a video file to detect if it's real or fake.</p> */}
      </div>
      <Container>
        <h1 className="title">Deepfake Detection</h1>
        <Form className="form" onSubmit={handleSubmit}>
          <Form.Group controlId="formFile">
            <Form.Label className="form-label">Upload Video</Form.Label>
            <Form.Control type="file" onChange={handleFileChange} className="form-control" />
          </Form.Group>
          {videoPreview && (
          <div className="video-preview">
            <p className="video-preview-title">Video Preview:</p>
            <video controls src={videoPreview} className="video-preview-player" />
          </div>
        )}
          <div className="submit-group">
            <Button variant="primary" type="submit" disabled={loading} className={loading ? 'loading' : ''}>
              {loading && <><div className="spinner" /> Analyzing</>}
              {!loading && "Submit"}
            </Button>
          </div>
        </Form>
        
        {prediction !== null && (
          <div className="result">
            <h2 className="result-title">Prediction: {prediction === 1 ? 'Real' : 'Fake'}</h2>
            <p className="result-confidence">Confidence: {confidence.toFixed(2)}%</p>
            {image && <Image src={image} alt="Heatmap" fluid className="result-image" />}
          </div>
        )}
      </Container>
    </>
  );
};

export default App;

