// React component example
import React, { useState } from 'react';

function ImageClassifier() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState('');

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onloadend = () => {
      setImage(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('image', image);

    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    setPrediction(data.prediction);
  };

  return (
    <div>
      <h1>Image Classifier</h1>
      <input type="file" onChange={handleImageUpload} />
      <button onClick={handleSubmit}>Classify</button>
      <p>{prediction}</p>
      {image && <img src={image} alt="Uploaded Image" />}
    </div>
  );
}
