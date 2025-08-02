import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
from data_processor import DataProcessor
from visualizations import Visualizations

# Page configuration
st.set_page_config(
    page_title="Financial Transaction Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Header with copyright and tech stack
    st.title("💰 Financial Transaction Analyzer")
    st.markdown("### Comprehensive Analysis of Your Daily Transactions")
    
    # Tech Stack Information
    with st.expander("🛠️ Tech Stack & Features"):
        st.markdown("""
        **Technology Stack:**
        - **Frontend Framework:** Streamlit (Python-based web framework)
        - **Data Processing:** Pandas & NumPy for efficient data manipulation
        - **Visualizations:** Plotly (Interactive charts and graphs)
        - **Backend:** Python 3.11 with advanced analytics
        - **Deployment:** Replit Cloud Platform
        
        **Advanced Features:**
        - ✅ Interactive data filtering and search
        - ✅ Fraud detection algorithms
        - ✅ Expense anomaly analysis
        - ✅ Predictive spending insights
        - ✅ Advanced statistical analysis
        - ✅ Multi-dimensional data visualization
        - ✅ Real-time data processing
        """)
    
    st.markdown("---")
    
    # Copyright notice
    st.sidebar.markdown("---")
    st.sidebar.markdown("© **Aditya Arora 2025**")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    
    # Sidebar for file upload and filters
    st.sidebar.header("📁 Data Upload")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload your transaction CSV file",
        type=['csv'],
        help="Upload a CSV file with transaction data containing columns: Date, Mode, Category, Subcategory, Note, Amount, Income/Expense, Currency"
    )
    
    # Load sample data option
    if st.sidebar.button("📊 Load Sample Data"):
        try:
            # Load the provided sample data
            sample_file_path = "attached_assets/Daily Household Transactions_1754146092697.csv"
            st.session_state.df = pd.read_csv(sample_file_path)
            st.session_state.data_loaded = True
            st.sidebar.success("Sample data loaded successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading sample data: {str(e)}")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.data_loaded = True
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
            st.session_state.data_loaded = False
    
    # Main content
    if st.session_state.data_loaded and st.session_state.df is not None:
        # Process data
        processor = DataProcessor()
        viz = Visualizations()
        
        try:
            st.session_state.processed_df = processor.clean_data(st.session_state.df)
            df = st.session_state.processed_df
            
            # Data validation
            if df.empty:
                st.error("The uploaded file appears to be empty or invalid.")
                return
            
            # Display data info
            st.sidebar.markdown("---")
            st.sidebar.header("📈 Data Overview")
            st.sidebar.metric("Total Transactions", len(df))
            st.sidebar.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            
            # Filters
            st.sidebar.header("🔍 Filters")
            
            # Date range filter
            min_date = df['Date'].min().date()
            max_date = df['Date'].max().date()
            
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
            
            # Category filter
            categories = ['All'] + sorted(df['Category'].unique().tolist())
            selected_category = st.sidebar.selectbox("Select Category", categories)
            
            if selected_category != 'All':
                df = df[df['Category'] == selected_category]
            
            # Payment mode filter
            modes = ['All'] + sorted(df['Mode'].unique().tolist())
            selected_mode = st.sidebar.selectbox("Select Payment Mode", modes)
            
            if selected_mode != 'All':
                df = df[df['Mode'] == selected_mode]
            
            # Transaction type filter
            transaction_types = ['All', 'Income', 'Expense']
            selected_type = st.sidebar.selectbox("Select Transaction Type", transaction_types)
            
            if selected_type != 'All':
                df = df[df['Income/Expense'] == selected_type]
            
            # Main dashboard
            display_dashboard(df, viz, processor)
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.info("Please ensure your CSV file has the required columns: Date, Mode, Category, Subcategory, Note, Amount, Income/Expense, Currency")
    
    else:
        # Welcome screen
        st.info("👆 Please upload a CSV file or load sample data to begin analysis")
        
        # Display expected format
        st.markdown("### 📋 Expected CSV Format")
        expected_format = pd.DataFrame({
            'Date': ['20/09/2018 12:04:08', '19/09/2018'],
            'Mode': ['Cash', 'Saving Bank account 1'],
            'Category': ['Transportation', 'subscription'],
            'Subcategory': ['Train', 'Netflix'],
            'Note': ['2 Place 5 to Place 0', '1 month subscription'],
            'Amount': [30.0, 199.0],
            'Income/Expense': ['Expense', 'Expense'],
            'Currency': ['INR', 'INR']
        })
        
        st.dataframe(expected_format, use_container_width=True)

def display_dashboard(df, viz, processor):
    """Display the main dashboard with all visualizations and metrics"""
    
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Key metrics
    st.header("📊 Financial Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_income = df[df['Income/Expense'] == 'Income']['Amount'].sum()
    total_expense = df[df['Income/Expense'] == 'Expense']['Amount'].sum()
    net_savings = total_income - total_expense
    avg_transaction = df['Amount'].mean()
    
    with col1:
        st.metric("💰 Total Income", f"₹{total_income:,.2f}")
    
    with col2:
        st.metric("💸 Total Expenses", f"₹{total_expense:,.2f}")
    
    with col3:
        st.metric("💵 Net Savings", f"₹{net_savings:,.2f}", 
                 delta=f"₹{net_savings:,.2f}" if net_savings >= 0 else f"-₹{abs(net_savings):,.2f}")
    
    with col4:
        st.metric("📈 Avg Transaction", f"₹{avg_transaction:,.2f}")
    
    st.markdown("---")
    
    # Advanced Analytics Section
    st.header("🔍 Advanced Analytics")
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚨 Fraud Detection", "📈 Expense Analysis", "🎯 Insights"])
    
    with tab1:
        # Row 1: Income vs Expense and Category Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Income vs Expenses")
            income_expense_fig = viz.create_income_expense_chart(df)
            st.plotly_chart(income_expense_fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Spending by Category")
            category_fig = viz.create_category_pie_chart(df)
            st.plotly_chart(category_fig, use_container_width=True)
        
        # Row 2: Monthly trends and Payment modes
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Monthly Trends")
            monthly_fig = viz.create_monthly_trends(df)
            st.plotly_chart(monthly_fig, use_container_width=True)
        
        with col2:
            st.subheader("💳 Payment Mode Distribution")
            payment_fig = viz.create_payment_mode_chart(df)
            st.plotly_chart(payment_fig, use_container_width=True)
        
        # Row 3: Top categories and daily trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔝 Top Spending Categories")
            top_categories_fig = viz.create_top_categories_chart(df)
            st.plotly_chart(top_categories_fig, use_container_width=True)
        
        with col2:
            st.subheader("📅 Daily Spending Pattern")
            daily_fig = viz.create_daily_pattern(df)
            st.plotly_chart(daily_fig, use_container_width=True)
    
    with tab2:
        # Fraud Detection Section
        st.subheader("🚨 Fraud Detection & Anomaly Analysis")
        
        # Detect potential fraudulent transactions
        fraud_results = detect_fraud_transactions(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🚨 Potential Fraud Transactions", len(fraud_results['suspicious_transactions']))
            st.metric("⚠️ High-Risk Patterns", fraud_results['risk_patterns'])
        
        with col2:
            st.metric("📊 Anomaly Score", f"{fraud_results['anomaly_score']:.2f}")
            st.metric("🎯 Risk Level", fraud_results['risk_level'])
        
        # Display suspicious transactions
        if len(fraud_results['suspicious_transactions']) > 0:
            st.subheader("🔍 Suspicious Transactions")
            st.dataframe(fraud_results['suspicious_transactions'], use_container_width=True)
        
        # Fraud detection visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount distribution with outliers
            outlier_fig = viz.create_outlier_detection_chart(df)
            st.plotly_chart(outlier_fig, use_container_width=True)
        
        with col2:
            # Time-based anomaly detection
            time_anomaly_fig = viz.create_time_anomaly_chart(df)
            st.plotly_chart(time_anomaly_fig, use_container_width=True)
    
    with tab3:
        # Expense Analysis Section
        st.subheader("📈 Advanced Expense Analysis")
        
        # Highest and Lowest Expenses
        expense_df = df[df['Income/Expense'] == 'Expense']
        
        if not expense_df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Highest expenses
                highest_expenses = expense_df.nlargest(5, 'Amount')
                st.subheader("🔥 Highest Expenses")
                for idx, row in highest_expenses.iterrows():
                    st.write(f"**₹{row['Amount']:,.2f}** - {row['Category']} ({row['Date'].strftime('%Y-%m-%d')})")
            
            with col2:
                # Lowest expenses
                lowest_expenses = expense_df.nsmallest(5, 'Amount')
                st.subheader("❄️ Lowest Expenses")
                for idx, row in lowest_expenses.iterrows():
                    st.write(f"**₹{row['Amount']:,.2f}** - {row['Category']} ({row['Date'].strftime('%Y-%m-%d')})")
            
            with col3:
                # Expense statistics
                st.subheader("📊 Expense Statistics")
                st.metric("Average Expense", f"₹{expense_df['Amount'].mean():,.2f}")
                st.metric("Median Expense", f"₹{expense_df['Amount'].median():,.2f}")
                st.metric("Std Deviation", f"₹{expense_df['Amount'].std():,.2f}")
        
        # Advanced expense visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot for expense distribution
            expense_box_fig = viz.create_expense_box_plot(df)
            st.plotly_chart(expense_box_fig, use_container_width=True)
        
        with col2:
            # Heatmap of spending patterns
            spending_heatmap_fig = viz.create_spending_heatmap(df)
            st.plotly_chart(spending_heatmap_fig, use_container_width=True)
        
        # Expense trend analysis
        expense_trend_fig = viz.create_expense_trend_analysis(df)
        st.plotly_chart(expense_trend_fig, use_container_width=True)
    
    with tab4:
        # Enhanced Insights and Recommendations
        st.subheader("🎯 Financial Insights & Strategic Recommendations")
        
        # Generate comprehensive insights
        insights = generate_financial_insights(df)
        
        # Key Financial Metrics Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💡 Key Financial Insights")
            for insight in insights['key_insights']:
                st.info(insight)
        
        with col2:
            st.subheader("💰 Savings Optimization")
            if insights['savings_optimization']:
                for strategy in insights['savings_optimization']:
                    if "Urgent" in strategy or "overspending" in strategy:
                        st.error(strategy)
                    else:
                        st.success(strategy)
            else:
                st.info("No specific savings optimization available")
        
        st.markdown("---")
        
        # Expense Reduction Strategies
        if insights['expense_reduction_strategies']:
            st.subheader("📉 Expense Reduction Strategies")
            st.markdown("**How to reduce expenses below income:**")
            
            for i, strategy in enumerate(insights['expense_reduction_strategies'], 1):
                st.markdown(f"**{i}.** {strategy}")
        
        # Smart Recommendations
        if insights['recommendations']:
            st.subheader("🧠 Smart Recommendations")
            st.markdown("**Personalized suggestions based on your spending patterns:**")
            
            for i, recommendation in enumerate(insights['recommendations'], 1):
                st.success(f"💡 {recommendation}")
        
        # Budget Control Section
        if insights['budget_alerts']:
            st.subheader("⚡ Budget Control & Alerts")
            st.markdown("**Recommended budget limits to control spending:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Monthly Category Budgets:**")
                for alert in insights['budget_alerts'][:-1]:  # All except weekly limit
                    st.warning(alert)
            
            with col2:
                st.markdown("**Weekly Spending Control:**")
                if insights['budget_alerts']:
                    st.warning(insights['budget_alerts'][-1])  # Weekly limit
                
                # Additional quick tips
                st.markdown("**Quick Tips:**")
                st.markdown("• Use the 50-30-20 rule: 50% needs, 30% wants, 20% savings")
                st.markdown("• Track daily expenses to stay within budget")
                st.markdown("• Set up automatic savings transfers")
                st.markdown("• Review and cancel unused subscriptions monthly")
        
        st.markdown("---")
        
        # Action Plan Section
        st.subheader("📋 30-Day Action Plan")
        
        # Calculate potential savings
        total_income = df[df['Income/Expense'] == 'Income']['Amount'].sum()
        total_expense = df[df['Income/Expense'] == 'Expense']['Amount'].sum()
        current_savings_rate = ((total_income - total_expense) / total_income * 100) if total_income > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Week 1-2: Analysis**")
            st.markdown("✅ Track all expenses daily")
            st.markdown("✅ Identify top 3 spending categories")
            st.markdown("✅ Set up digital payment tracking")
        
        with col2:
            st.markdown("**Week 3: Implementation**")
            st.markdown("🎯 Reduce top category by 15%")
            st.markdown("🎯 Cancel unused subscriptions")
            st.markdown("🎯 Set weekly spending limits")
        
        with col3:
            st.markdown("**Week 4: Optimization**")
            st.markdown("💡 Compare with previous month")
            st.markdown("💡 Adjust budget based on results")
            st.markdown("💡 Plan next month's budget")
        
        # Predictive analysis
        st.subheader("📈 Spending Predictions & Targets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_fig = viz.create_spending_prediction(df)
            st.plotly_chart(prediction_fig, use_container_width=True)
        
        with col2:
            # Savings target visualization
            target_fig = viz.create_savings_target_chart(df)
            st.plotly_chart(target_fig, use_container_width=True)
    
    # Transaction details table
    st.markdown("---")
    st.header("📋 Transaction Details")
    
    # Search functionality
    search_term = st.text_input("🔍 Search transactions (by note or category)")
    
    if search_term:
        mask = (df['Note'].str.contains(search_term, case=False, na=False) | 
                df['Category'].str.contains(search_term, case=False, na=False) |
                df['Subcategory'].str.contains(search_term, case=False, na=False))
        filtered_df = df[mask]
    else:
        filtered_df = df
    
    # Display filtered data
    st.dataframe(
        filtered_df[['Date', 'Category', 'Subcategory', 'Note', 'Amount', 'Income/Expense', 'Mode']].sort_values('Date', ascending=False),
        use_container_width=True
    )
    
    # Export functionality
    st.markdown("---")
    st.header("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export summary statistics
        summary_stats = {
            'Total Income': total_income,
            'Total Expenses': total_expense,
            'Net Savings': net_savings,
            'Average Transaction': avg_transaction,
            'Total Transactions': len(filtered_df),
            'Date Range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
        }
        
        summary_df = pd.DataFrame(list(summary_stats.items()))
        summary_df.columns = ['Metric', 'Value']
        summary_csv = summary_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Summary Statistics",
            data=summary_csv,
            file_name=f"transaction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def detect_fraud_transactions(df):
    """Detect potentially fraudulent transactions using multiple algorithms"""
    from scipy import stats
    
    suspicious_transactions = []
    risk_patterns = 0
    
    # 1. Statistical outlier detection using Z-score
    if len(df) > 10:
        amounts = df['Amount'].astype(float)
        z_scores = np.abs(stats.zscore(amounts))
        outliers = df[z_scores > 3]  # Transactions with Z-score > 3
        suspicious_transactions.extend(outliers.to_dict('records'))
        
    # 2. Time-based anomaly detection
    df_sorted = df.sort_values('Date')
    df_sorted['Hour'] = df_sorted['Date'].dt.hour
    
    # Unusual time patterns (transactions at odd hours)
    unusual_times = df_sorted[(df_sorted['Hour'] < 6) | (df_sorted['Hour'] > 23)]
    if len(unusual_times) > 0:
        risk_patterns += 1
        suspicious_transactions.extend(unusual_times.to_dict('records'))
    
    # 3. Amount-based patterns
    # Very high amounts compared to user's average
    avg_amount = df['Amount'].mean()
    std_amount = df['Amount'].std()
    high_amount_threshold = avg_amount + (3 * std_amount)
    
    high_amounts = df[df['Amount'] > high_amount_threshold]
    if len(high_amounts) > 0:
        risk_patterns += 1
        suspicious_transactions.extend(high_amounts.to_dict('records'))
    
    # 4. Frequency-based detection
    # Multiple transactions on the same day with same amount
    daily_duplicates = df.groupby([df['Date'].dt.date, 'Amount']).size()
    duplicate_transactions = daily_duplicates[daily_duplicates > 3]
    if len(duplicate_transactions) > 0:
        risk_patterns += 1
    
    # Remove duplicates from suspicious transactions
    seen = set()
    unique_suspicious = []
    for transaction in suspicious_transactions:
        # Create a unique identifier for each transaction
        identifier = f"{transaction['Date']}_{transaction['Amount']}_{transaction['Category']}"
        if identifier not in seen:
            seen.add(identifier)
            unique_suspicious.append(transaction)
    
    # Calculate anomaly score
    total_transactions = len(df)
    anomaly_score = len(unique_suspicious) / total_transactions * 100 if total_transactions > 0 else 0
    
    # Determine risk level
    if anomaly_score > 10:
        risk_level = "🔴 High"
    elif anomaly_score > 5:
        risk_level = "🟡 Medium"
    else:
        risk_level = "🟢 Low"
    
    return {
        'suspicious_transactions': pd.DataFrame(unique_suspicious),
        'risk_patterns': risk_patterns,
        'anomaly_score': anomaly_score,
        'risk_level': risk_level
    }

def generate_financial_insights(df):
    """Generate comprehensive financial insights and recommendations"""
    insights = {
        'key_insights': [],
        'recommendations': [],
        'expense_reduction_strategies': [],
        'savings_optimization': [],
        'budget_alerts': []
    }
    
    # Calculate basic metrics
    total_income = df[df['Income/Expense'] == 'Income']['Amount'].sum()
    total_expense = df[df['Income/Expense'] == 'Expense']['Amount'].sum()
    net_savings = total_income - total_expense
    
    # Expense analysis
    expense_df = df[df['Income/Expense'] == 'Expense']
    if not expense_df.empty:
        # Category analysis
        category_spending = expense_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        top_category = category_spending.index[0]
        top_category_amount = category_spending.iloc[0]
        
        # Time-based analysis
        monthly_expenses = expense_df.groupby(expense_df['Date'].dt.to_period('M'))['Amount'].sum()
        avg_monthly_expense = monthly_expenses.mean()
        
        # Payment mode analysis
        payment_analysis = expense_df.groupby('Mode')['Amount'].sum().sort_values(ascending=False)
        
        # Generate insights
        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
        expense_ratio = (total_expense / total_income * 100) if total_income > 0 else 0
        
        insights['key_insights'].append(f"Your savings rate is {savings_rate:.1f}% (Ideal: 20%+)")
        insights['key_insights'].append(f"Your expense ratio is {expense_ratio:.1f}% of income")
        insights['key_insights'].append(f"Highest spending: '{top_category}' (₹{top_category_amount:,.2f})")
        insights['key_insights'].append(f"Average monthly expenses: ₹{avg_monthly_expense:,.2f}")
        
        # Generate expense reduction strategies
        if savings_rate < 20:
            target_reduction = (20 - savings_rate) / 100 * total_income
            insights['expense_reduction_strategies'].append(f"Target reduction needed: ₹{target_reduction:,.2f} to achieve 20% savings rate")
            
            # Category-specific recommendations
            for i, (category, amount) in enumerate(category_spending.head(3).items()):
                reduction_pct = 15 if i == 0 else 10  # More reduction from top category
                potential_savings = amount * (reduction_pct / 100)
                insights['expense_reduction_strategies'].append(
                    f"Reduce '{category}' by {reduction_pct}%: Save ₹{potential_savings:,.2f}/month"
                )
        
        # Savings optimization strategies
        if net_savings > 0:
            insights['savings_optimization'].append(f"Great! You're saving ₹{net_savings:,.2f}. Consider investing in mutual funds or FDs")
            
            if savings_rate > 30:
                insights['savings_optimization'].append("Excellent savings rate! You could explore higher-return investments")
        else:
            deficit = abs(net_savings)
            insights['savings_optimization'].append(f"Urgent: You're overspending by ₹{deficit:,.2f}. Immediate expense reduction needed")
            
        # Smart recommendations based on spending patterns
        recommendations = []
        
        # High expense categories
        if top_category_amount > total_expense * 0.4:
            recommendations.append(f"'{top_category}' is 40%+ of expenses. Break it down into subcategories for better control")
            
        # Food category specific
        if 'Food' in category_spending.index and category_spending['Food'] > avg_monthly_expense * 0.3:
            recommendations.append("Food expenses are high. Try meal planning, cooking at home, and bulk buying")
            
        # Transportation specific
        if 'Transportation' in category_spending.index and category_spending['Transportation'] > avg_monthly_expense * 0.2:
            recommendations.append("Consider carpooling, public transport, or fuel-efficient travel to reduce transportation costs")
            
        # Entertainment/subscriptions
        entertainment_cats = ['subscription', 'Culture', 'Entertainment']
        entertainment_total = sum(category_spending.get(cat, 0) for cat in entertainment_cats)
        if entertainment_total > avg_monthly_expense * 0.15:
            recommendations.append("Review subscriptions and entertainment expenses. Cancel unused services")
            
        # Payment mode optimization
        cash_spending = payment_analysis.get('Cash', 0)
        if cash_spending > total_expense * 0.5:
            recommendations.append("High cash usage detected. Switch to digital payments for better tracking and cashback")
            
        # Frequency-based recommendations
        frequent_small_purchases = expense_df[expense_df['Amount'] < 100]
        if len(frequent_small_purchases) > len(expense_df) * 0.6:
            recommendations.append("Many small purchases detected. Set daily spending limits to control impulse buying")
            
        # Budget alerts and controls
        budget_alerts = []
        for category, amount in category_spending.head(5).items():
            monthly_limit = amount * 0.9  # 10% reduction target
            budget_alerts.append(f"Set monthly budget for '{category}': ₹{monthly_limit:,.0f}")
            
        # Weekly spending limit
        weekly_limit = avg_monthly_expense / 4 * 0.9  # 10% reduction
        budget_alerts.append(f"Recommended weekly spending limit: ₹{weekly_limit:,.0f}")
        
        # Additional expense reduction tips
        general_tips = [
            "Use the envelope method: Allocate cash for each category and stop when it's gone",
            "Implement the 24-hour rule: Wait a day before making non-essential purchases",
            "Buy generic brands instead of premium ones to save 20-30%",
            "Use apps to compare prices and find deals before shopping",
            "Set up automatic transfers to savings account on payday",
            "Review and negotiate recurring bills (insurance, phone, internet)",
            "Use public transportation or walk instead of driving when possible",
            "Pack meals from home instead of eating out frequently"
        ]
        
        # Add general tips if specific recommendations are low
        if len(recommendations) < 3:
            recommendations.extend(general_tips[:3])
        
        insights['recommendations'] = recommendations
        insights['budget_alerts'] = budget_alerts
        
    return insights

if __name__ == "__main__":
    main()
