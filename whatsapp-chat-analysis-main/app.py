import streamlit as st
import preprocessor,helper
import sentiment
import matplotlib.pyplot as plt
import seaborn as sns
from emoji import EMOJI_DATA

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
            
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        if user_heatmap is not None and not user_heatmap.empty:
            fig,ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)
        else:
            st.info("Not enough data to display the activity heatmap.")

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        if df_wc is not None:
            fig,ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)
        else:
            st.info("Not enough words to generate a word cloud.")


        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)

        # Extracting and displaying emojis from messages
        st.title("Emoji Extractor")
        if selected_user == 'Overall':
            st.warning("Emoji extraction is not available for overall data.")
        else:
            message = df[df['user'] == selected_user]['message']
            emojis = []
            for msg in message:
                emojis.extend([c for c in msg if c in EMOJI_DATA])
            
            if emojis:
                st.write(" ".join(emojis))
            else:
                st.info("No emojis found in the selected user's messages.")

        # Sentiment Analysis Section
        st.title("📊 Sentiment Analysis")
        
        # Perform sentiment analysis
        sentiment_df = sentiment.comprehensive_sentiment_analysis(selected_user, df)
        
        if sentiment_df is not None and not sentiment_df.empty:
            # Get sentiment summary
            summary = sentiment.get_sentiment_summary(sentiment_df)
            
            if summary:
                # Display sentiment metrics
                st.subheader("Sentiment Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Positive Messages", summary['positive_messages'])
                with col2:
                    st.metric("Negative Messages", summary['negative_messages'])
                with col3:
                    st.metric("Neutral Messages", summary['neutral_messages'])
                with col4:
                    st.metric("Avg Polarity", f"{summary['avg_polarity']:.3f}")
                
                # Add confidence metrics
                confidence_analysis = sentiment.get_confidence_analysis(sentiment_df)
                if confidence_analysis:
                    st.subheader("🤖 ML Model Confidence")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Avg Confidence", f"{confidence_analysis['avg_confidence']:.3f}")
                    with col2:
                        st.metric("High Confidence", confidence_analysis['high_confidence'])
                    with col3:
                        st.metric("Medium Confidence", confidence_analysis['medium_confidence'])
                    with col4:
                        st.metric("Low Confidence", confidence_analysis['low_confidence'])
                
                # Risk assessment
                st.subheader("🔍 Safety Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Dangerous Messages", summary['dangerous_messages'])
                with col2:
                    risk_color = "red" if summary['risk_level'] == "HIGH RISK" else "orange" if summary['risk_level'] == "MEDIUM RISK" else "green"
                    st.markdown(f"**Risk Level:** <span style='color: {risk_color}'>{summary['risk_level']}</span>", unsafe_allow_html=True)
                
                # Sentiment timeline
                st.subheader("📈 Sentiment Timeline")
                timeline_fig = sentiment.plot_sentiment_timeline(sentiment_df, selected_user)
                if timeline_fig:
                    st.pyplot(timeline_fig)
                
                # Danger analysis
                st.subheader("⚠️ Safety Analysis")
                danger_fig = sentiment.plot_danger_analysis(sentiment_df)
                if danger_fig:
                    st.pyplot(danger_fig)
                
                # Most dangerous messages
                dangerous_messages = sentiment.get_dangerous_messages(sentiment_df, limit=5)
                if not dangerous_messages.empty:
                    st.subheader("🚨 Potentially Concerning Messages")
                    st.dataframe(dangerous_messages, use_container_width=True)
                else:
                    st.success("✅ No concerning messages detected!")
                
                # Keyword analysis
                st.subheader("🔤 Keyword Analysis by Sentiment")
                keyword_analysis = sentiment.get_keyword_analysis(sentiment_df)
                
                if keyword_analysis:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Positive Keywords:**")
                        if keyword_analysis.get('positive'):
                            for word, count in keyword_analysis['positive'][:5]:
                                st.write(f"• {word} ({count})")
                        else:
                            st.write("No positive keywords found")
                    
                    with col2:
                        st.write("**Negative Keywords:**")
                        if keyword_analysis.get('negative'):
                            for word, count in keyword_analysis['negative'][:5]:
                                st.write(f"• {word} ({count})")
                        else:
                            st.write("No negative keywords found")
                    
                    with col3:
                        st.write("**Neutral Keywords:**")
                        if keyword_analysis.get('neutral'):
                            for word, count in keyword_analysis['neutral'][:5]:
                                st.write(f"• {word} ({count})")
                        else:
                            st.write("No neutral keywords found")
        else:
            st.warning("⚠️ Not enough data available for sentiment analysis. Please ensure your chat file contains text messages.")