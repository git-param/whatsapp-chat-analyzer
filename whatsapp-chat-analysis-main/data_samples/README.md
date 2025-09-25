# WhatsApp Chat Analysis - Data Samples

This directory contains various sample WhatsApp chat files for testing the sentiment analysis functionality. Each file represents different types of conversations and sentiment patterns.

## Sample Files Description

### 1. **positive_chat.txt**
- **Content**: Highly positive conversation with enthusiastic team members
- **Sentiment**: Predominantly positive
- **Features**: Lots of positive emojis, encouraging messages, team appreciation
- **Expected ML Results**: High positive sentiment, high confidence scores

### 2. **negative_chat.txt**
- **Content**: Frustrated team members expressing dissatisfaction
- **Sentiment**: Predominantly negative
- **Features**: Complaints, frustration, disappointment
- **Expected ML Results**: High negative sentiment, medium confidence scores

### 3. **neutral_chat.txt**
- **Content**: Professional, factual conversation
- **Sentiment**: Neutral/informational
- **Features**: Business-like communication, no emotional expressions
- **Expected ML Results**: Neutral sentiment, medium confidence scores

### 4. **mixed_sentiment_chat.txt**
- **Content**: Conversation with varying emotions and opinions
- **Sentiment**: Mixed (positive, negative, neutral)
- **Features**: Different team members with different emotional states
- **Expected ML Results**: Balanced sentiment distribution, varied confidence

### 5. **dangerous_content_chat.txt**
- **Content**: Conversation containing threats and concerning language
- **Sentiment**: Negative with dangerous content
- **Features**: Threats, violence, harassment
- **Expected ML Results**: High negative sentiment, HIGH RISK level, flagged dangerous messages

### 6. **emoji_heavy_chat.txt**
- **Content**: Conversation with excessive emoji usage
- **Sentiment**: Very positive
- **Features**: Heavy emoji usage, enthusiastic expressions
- **Expected ML Results**: High positive sentiment, high confidence

### 7. **business_meeting_chat.txt**
- **Content**: Professional business meeting discussion
- **Sentiment**: Neutral/professional
- **Features**: Formal language, business terminology, factual information
- **Expected ML Results**: Neutral sentiment, medium confidence

### 8. **family_chat.txt**
- **Content**: Warm family conversation
- **Sentiment**: Positive and caring
- **Features**: Family bonding, care, support
- **Expected ML Results**: Positive sentiment, high confidence

### 9. **student_group_chat.txt**
- **Content**: Student study group conversation
- **Sentiment**: Positive and supportive
- **Features**: Academic support, friendship, encouragement
- **Expected ML Results**: Positive sentiment, high confidence

### 10. **minimal_chat.txt**
- **Content**: Very short, minimal conversation
- **Sentiment**: Neutral
- **Features**: Short responses, basic greetings
- **Expected ML Results**: Neutral sentiment, low confidence (due to limited text)

## Usage Instructions

1. **Upload any of these files** to the Streamlit app for testing
2. **Compare results** across different file types to see how the ML model performs
3. **Test individual users** vs overall analysis for each file
4. **Verify safety analysis** works correctly with the dangerous content file
5. **Check confidence scores** vary appropriately with content complexity

## Expected ML Model Performance

- **High Confidence**: Files with clear emotional language (positive_chat.txt, negative_chat.txt)
- **Medium Confidence**: Professional or mixed content (business_meeting_chat.txt, mixed_sentiment_chat.txt)
- **Low Confidence**: Minimal content (minimal_chat.txt)
- **Risk Detection**: Should flag dangerous_content_chat.txt as HIGH RISK

## Testing Scenarios

1. **Sentiment Accuracy**: Test if the model correctly identifies sentiment patterns
2. **Confidence Scoring**: Verify confidence levels match content clarity
3. **Safety Analysis**: Ensure dangerous content is properly flagged
4. **User Analysis**: Test individual user sentiment vs overall group sentiment
5. **Visualization**: Check if charts and graphs display correctly for each file type

## File Format

All files follow the standard WhatsApp export format:
```
DD/MM/YYYY, HH:MM - Username: Message content
```

This format is compatible with the preprocessor.py module and will work seamlessly with the sentiment analysis system.

