import React, { useState } from 'react';
import './Dashboard.css';
import Button from './Button';

const CustomerReviews = () => {
  const [reviews, setReviews] = useState([
    {
      id: 1,
      name: "Alex P.",
      text: "Blocked 12 phishing attempts this week!",
      rating: 5,
      avatar: "/avatars/alex.jpg",
      liked: false,
      reported: false
    },
    {
      id: 2,
      name: "Sam R.",
      text: "Saved me from a fake tech support call.",
      rating: 4,
      avatar: "/avatars/sam.jpg",
      liked: false,
      reported: false
    }
  ]);

  const [newReview, setNewReview] = useState({
    name: '',
    text: '',
    rating: 5
  });

  const handleLike = (id) => {
    setReviews(reviews.map(review => 
      review.id === id ? {...review, liked: !review.liked} : review
    ));
  };

  const handleReport = (id) => {
    setReviews(reviews.map(review => 
      review.id === id ? {...review, reported: true} : review
    ));
    alert("Review reported. Our team will investigate.");
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewReview(prev => ({...prev, [name]: value}));
  };

  const handleSubmitReview = (e) => {
    e.preventDefault();
    if (newReview.name && newReview.text) {
      const review = {
        id: reviews.length + 1,
        name: newReview.name,
        text: newReview.text,
        rating: parseInt(newReview.rating),
        avatar: `/avatars/default.jpg`,
        liked: false,
        reported: false
      };
      setReviews([...reviews, review]);
      setNewReview({ name: '', text: '', rating: 5 });
    }
  };

  return (
    <div className="customer-reviews">
      <div className="reviews-header">
        <h3>User Reviews</h3>
        <Button className="see-all-btn">See All Reviews</Button>
      </div>
      
      <div className="reviews-grid">
        {reviews.map(review => (
          <div key={review.id} className="review-card">
            <div className="review-header">
              <img src={review.avatar} alt={review.name} />
              <h4>{review.name}</h4>
            </div>
            <p>"{review.text}"</p>
            <div className="stars">
              {'★'.repeat(review.rating)}{'☆'.repeat(5 - review.rating)}
            </div>
            <div className="review-actions">
              <Button 
                onClick={() => handleLike(review.id)}
                className={`like-btn ${review.liked ? 'liked' : ''}`}
              >
                {review.liked ? '❤️ Liked' : '🤍 Like'}
              </Button>
              <Button 
                onClick={() => handleReport(review.id)}
                disabled={review.reported}
                className="report-btn"
              >
                {review.reported ? 'Reported' : 'Report'}
              </Button>
            </div>
          </div>
        ))}
      </div>

      <div className="add-review">
        <h4>Add Your Review</h4>
        <form onSubmit={handleSubmitReview}>
          <input
            type="text"
            name="name"
            placeholder="Your Name"
            value={newReview.name}
            onChange={handleInputChange}
            required
          />
          <textarea
            name="text"
            placeholder="Your Review"
            value={newReview.text}
            onChange={handleInputChange}
            required
          />
          <div className="rating-select">
            <label>Rating:</label>
            <select 
              name="rating" 
              value={newReview.rating}
              onChange={handleInputChange}
            >
              {[5,4,3,2,1].map(num => (
                <option key={num} value={num}>{'★'.repeat(num)}{'☆'.repeat(5 - num)}</option>
              ))}
            </select>
          </div>
          <Button type="submit" className="submit-review-btn">
            Submit Review
          </Button>
        </form>
      </div>
    </div>
  );
};

export default CustomerReviews;