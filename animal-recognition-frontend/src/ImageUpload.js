import React, { useState } from 'react';
import axios from 'axios';
import './App.css';  // We'll create some CSS for styling

function ImageUpload() {
    const [image, setImage] = useState(null);  // Store the uploaded image
    const [preview, setPreview] = useState('');  // Store the preview URL
    const [result, setResult] = useState('');  // Store the classification result
    const [loading, setLoading] = useState(false);  // Show loading indicator

    // Handle image change and preview setup
    const handleImageChange = (e) => {
        const file = e.target.files[0];
        setImage(file);
        setPreview(URL.createObjectURL(file));
        setResult('');  // Reset result when a new image is uploaded
    };

    // Submit image for classification
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);  // Start loading indicator
    
        // Debugging: Check if image state contains the correct file
        console.log("Image being uploaded:", image);
    
        const formData = new FormData();
        formData.append('image', image);
    
        try {
            const response = await axios.post('http://127.0.0.1:5000/classify', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setResult(response.data.classification);  // Set result to the classification result
        } catch (error) {
            // Log the full error object to check the issue
            console.error('Error uploading image:', error);
    
            // Check if there's a specific error message from Axios
            if (error.response) {
                console.log('Server responded with status:', error.response.status);
                console.log('Response data:', error.response.data);
            }
    
            setResult('Error occurred during classification.');
        } finally {
            setLoading(false);  // Stop loading indicator
        }
    };
    

    return (
        <div className="image-upload-container">
            <h1>Animal Image Recognition</h1>
            <form onSubmit={handleSubmit} className="upload-form">
                <div className="file-input">
                    <input type="file" accept="image/*" onChange={handleImageChange} required />
                    {preview && <img src={preview} alt="Preview" className="image-preview" />}
                </div>
                <button type="submit" disabled={!image || loading} className="upload-button">
                    {loading ? 'Classifying...' : 'Upload & Classify'}
                </button>
            </form>
            {loading && <div className="loading-spinner">Processing...</div>}
            {result && <h2 className="result">Result: {result}</h2>}
        </div>
    );
}

export default ImageUpload;
