import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizations:
    """Class to create various visualizations for transaction data"""
    
    def __init__(self):
        # Color schemes
        self.colors = {
            'income': '#2E8B57',      # Sea Green
            'expense': '#DC143C',     # Crimson
            'primary': '#1f77b4',     # Blue
            'secondary': '#ff7f0e',   # Orange
            'success': '#2ca02c',     # Green
            'warning': '#d62728',     # Red
        }
        
        self.color_palette = px.colors.qualitative.Set3
    
    def create_income_expense_chart(self, df):
        """Create income vs expense comparison chart"""
        # Aggregate data by type
        summary = df.groupby('Income/Expense')['Amount'].sum().reset_index()
        
        fig = go.Figure()
        
        # Add bars
        for idx, row in summary.iterrows():
            color = self.colors['income'] if row['Income/Expense'] == 'Income' else self.colors['expense']
            fig.add_trace(go.Bar(
                x=[row['Income/Expense']],
                y=[row['Amount']],
                name=row['Income/Expense'],
                marker_color=color,
                text=[f"₹{row['Amount']:,.2f}"],
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Income vs Expenses Comparison",
            xaxis_title="Transaction Type",
            yaxis_title="Amount (₹)",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_category_pie_chart(self, df):
        """Create pie chart for spending by category (expenses only)"""
        # Filter for expenses only
        expense_df = df[df['Income/Expense'] == 'Expense']
        
        if expense_df.empty:
            # Return empty chart if no expenses
            fig = go.Figure()
            fig.add_annotation(
                text="No expense data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
        
        # Group by category
        category_data = expense_df.groupby('Category')['Amount'].sum().reset_index()
        category_data = category_data.sort_values('Amount', ascending=False)
        
        # Take top 10 categories and group rest as 'Others'
        if len(category_data) > 10:
            top_categories = category_data.head(10)
            others_amount = category_data.tail(len(category_data) - 10)['Amount'].sum()
            others_row = pd.DataFrame({'Category': ['Others'], 'Amount': [others_amount]})
            category_data = pd.concat([top_categories, others_row], ignore_index=True)
        
        fig = px.pie(
            category_data, 
            values='Amount', 
            names='Category',
            title="Spending Distribution by Category",
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Amount: ₹%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_monthly_trends(self, df):
        """Create monthly trends chart"""
        # Group by month and transaction type
        monthly_data = df.groupby(['Month_Year', 'Income/Expense'])['Amount'].sum().reset_index()
        monthly_data['Month_Year_Str'] = monthly_data['Month_Year'].astype(str)
        
        fig = go.Figure()
        
        # Add traces for income and expense
        for transaction_type in ['Income', 'Expense']:
            type_data = monthly_data[monthly_data['Income/Expense'] == transaction_type]
            
            color = self.colors['income'] if transaction_type == 'Income' else self.colors['expense']
            
            fig.add_trace(go.Scatter(
                x=type_data['Month_Year_Str'],
                y=type_data['Amount'],
                mode='lines+markers',
                name=transaction_type,
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>{transaction_type}</b><br>Month: %{{x}}<br>Amount: ₹%{{y:,.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Monthly Income vs Expenses Trend",
            xaxis_title="Month",
            yaxis_title="Amount (₹)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_payment_mode_chart(self, df):
        """Create payment mode distribution chart"""
        mode_data = df.groupby('Mode')['Amount'].sum().reset_index()
        mode_data = mode_data.sort_values('Amount', ascending=True)
        
        fig = px.bar(
            mode_data, 
            x='Amount', 
            y='Mode',
            orientation='h',
            title="Transaction Amount by Payment Mode",
            color='Amount',
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Amount: ₹%{x:,.2f}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis_title="Amount (₹)",
            yaxis_title="Payment Mode",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_top_categories_chart(self, df):
        """Create top spending categories chart"""
        # Filter for expenses only
        expense_df = df[df['Income/Expense'] == 'Expense']
        
        if expense_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No expense data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
        
        # Get top 10 categories
        top_categories = expense_df.groupby('Category')['Amount'].sum().sort_values(ascending=True).tail(10)
        
        fig = px.bar(
            x=top_categories.values,
            y=top_categories.index,
            orientation='h',
            title="Top 10 Spending Categories",
            color=top_categories.values,
            color_continuous_scale='Reds'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Amount: ₹%{x:,.2f}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis_title="Amount (₹)",
            yaxis_title="Category",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_daily_pattern(self, df):
        """Create daily spending pattern chart"""
        # Group by weekday
        daily_data = df.groupby(['Weekday', 'Income/Expense'])['Amount'].sum().reset_index()
        
        # Define weekday order
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_data['Weekday'] = pd.Categorical(daily_data['Weekday'], categories=weekday_order, ordered=True)
        daily_data = daily_data.sort_values('Weekday')
        
        fig = go.Figure()
        
        # Add traces for income and expense
        for transaction_type in ['Income', 'Expense']:
            type_data = daily_data[daily_data['Income/Expense'] == transaction_type]
            
            color = self.colors['income'] if transaction_type == 'Income' else self.colors['expense']
            
            fig.add_trace(go.Bar(
                x=type_data['Weekday'],
                y=type_data['Amount'],
                name=transaction_type,
                marker_color=color,
                hovertemplate=f'<b>{transaction_type}</b><br>Day: %{{x}}<br>Amount: ₹%{{y:,.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Daily Spending Pattern by Weekday",
            xaxis_title="Day of Week",
            yaxis_title="Amount (₹)",
            height=400,
            barmode='group'
        )
        
        return fig
    
    def create_subcategory_analysis(self, df, category):
        """Create subcategory analysis for a specific category"""
        category_df = df[df['Category'] == category]
        
        if category_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {category}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
        
        subcategory_data = category_df.groupby('Subcategory')['Amount'].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=subcategory_data.index,
            y=subcategory_data.values,
            title=f"Subcategory Breakdown for {category}",
            color=subcategory_data.values,
            color_continuous_scale='Blues'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Amount: ₹%{y:,.2f}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis_title="Subcategory",
            yaxis_title="Amount (₹)",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_trend_comparison(self, df, metric='Amount'):
        """Create comparison trends over time"""
        # Monthly trends with multiple metrics
        monthly_trends = df.groupby(['Month_Year', 'Income/Expense']).agg({
            'Amount': 'sum',
            'Date': 'count'  # Transaction count
        }).reset_index()
        
        monthly_trends.rename(columns={'Date': 'Transaction_Count'}, inplace=True)
        monthly_trends['Month_Year_Str'] = monthly_trends['Month_Year'].astype(str)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Amount Trends', 'Transaction Count Trends'),
            vertical_spacing=0.1
        )
        
        # Add amount trends
        for transaction_type in ['Income', 'Expense']:
            type_data = monthly_trends[monthly_trends['Income/Expense'] == transaction_type]
            color = self.colors['income'] if transaction_type == 'Income' else self.colors['expense']
            
            fig.add_trace(
                go.Scatter(
                    x=type_data['Month_Year_Str'],
                    y=type_data['Amount'],
                    mode='lines+markers',
                    name=f'{transaction_type} Amount',
                    line=dict(color=color),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=type_data['Month_Year_Str'],
                    y=type_data['Transaction_Count'],
                    mode='lines+markers',
                    name=f'{transaction_type} Count',
                    line=dict(color=color, dash='dash'),
                    showlegend=True
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Monthly Trends Comparison",
            height=600
        )
        
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Amount (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Transaction Count", row=2, col=1)
        
        return fig
    
    def create_outlier_detection_chart(self, df):
        """Create outlier detection chart for fraud analysis"""
        from scipy import stats
        
        # Calculate Z-scores for amount outliers
        amounts = df['Amount'].astype(float)
        z_scores = np.abs(stats.zscore(amounts))
        df_with_zscore = df.copy()
        df_with_zscore['z_score'] = z_scores
        df_with_zscore['is_outlier'] = z_scores > 3
        
        # Create scatter plot
        fig = px.scatter(
            df_with_zscore,
            x=df_with_zscore.index,
            y='Amount',
            color='is_outlier',
            title="Transaction Amount Outlier Detection",
            labels={'x': 'Transaction Index', 'Amount': 'Amount (₹)'},
            color_discrete_map={True: 'red', False: 'blue'},
            hover_data=['Category', 'Date', 'z_score']
        )
        
        fig.update_layout(
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_time_anomaly_chart(self, df):
        """Create time-based anomaly detection chart"""
        # Group by hour of day
        df_with_hour = df.copy()
        df_with_hour['Hour'] = df_with_hour['Date'].dt.hour
        hourly_data = df_with_hour.groupby('Hour')['Amount'].agg(['count', 'sum']).reset_index()
        
        # Mark unusual hours (typically 11 PM to 6 AM)
        hourly_data['is_unusual'] = (hourly_data['Hour'] < 6) | (hourly_data['Hour'] > 23)
        
        fig = go.Figure()
        
        # Add bar chart for transaction counts
        fig.add_trace(go.Bar(
            x=hourly_data['Hour'],
            y=hourly_data['count'],
            name='Transaction Count',
            marker_color=['red' if unusual else 'blue' for unusual in hourly_data['is_unusual']],
            hovertemplate='Hour: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Transaction Pattern by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Transactions",
            height=400
        )
        
        return fig
    
    def create_expense_box_plot(self, df):
        """Create box plot for expense distribution analysis"""
        expense_df = df[df['Income/Expense'] == 'Expense']
        
        if expense_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No expense data available", x=0.5, y=0.5)
            return fig
        
        # Get top categories for box plot
        top_categories = expense_df.groupby('Category')['Amount'].sum().nlargest(8).index
        expense_filtered = expense_df[expense_df['Category'].isin(top_categories)]
        
        fig = px.box(
            expense_filtered,
            x='Category',
            y='Amount',
            title="Expense Distribution by Top Categories",
            color='Category',
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
    
    def create_spending_heatmap(self, df):
        """Create heatmap of spending patterns"""
        expense_df = df[df['Income/Expense'] == 'Expense']
        
        if expense_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No expense data available", x=0.5, y=0.5)
            return fig
        
        # Create month-category heatmap
        expense_df['Month_Name'] = expense_df['Date'].dt.strftime('%B')
        heatmap_data = expense_df.groupby(['Month_Name', 'Category'])['Amount'].sum().reset_index()
        
        # Pivot for heatmap
        heatmap_pivot = heatmap_data.pivot(index='Category', columns='Month_Name', values='Amount').fillna(0)
        
        # Select top categories to avoid overcrowding
        if len(heatmap_pivot) > 10:
            top_categories = expense_df.groupby('Category')['Amount'].sum().nlargest(10).index
            heatmap_pivot = heatmap_pivot.loc[top_categories]
        
        fig = px.imshow(
            heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            color_continuous_scale='Reds',
            title="Spending Heatmap: Category vs Month"
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_expense_trend_analysis(self, df):
        """Create advanced expense trend analysis"""
        expense_df = df[df['Income/Expense'] == 'Expense']
        
        if expense_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No expense data available", x=0.5, y=0.5)
            return fig
        
        # Monthly expense trends with moving average
        monthly_expenses = expense_df.groupby(expense_df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
        monthly_expenses['Date_str'] = monthly_expenses['Date'].astype(str)
        
        # Calculate 3-month moving average
        monthly_expenses['Moving_Avg'] = monthly_expenses['Amount'].rolling(window=3, min_periods=1).mean()
        
        fig = go.Figure()
        
        # Add actual expenses
        fig.add_trace(go.Scatter(
            x=monthly_expenses['Date_str'],
            y=monthly_expenses['Amount'],
            mode='lines+markers',
            name='Monthly Expenses',
            line=dict(color=self.colors['expense'], width=3),
            marker=dict(size=8)
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=monthly_expenses['Date_str'],
            y=monthly_expenses['Moving_Avg'],
            mode='lines',
            name='3-Month Moving Average',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Monthly Expense Trends with Moving Average",
            xaxis_title="Month",
            yaxis_title="Amount (₹)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_spending_prediction(self, df):
        """Create spending prediction chart using simple trend analysis"""
        expense_df = df[df['Income/Expense'] == 'Expense']
        
        if expense_df.empty or len(expense_df) < 3:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for prediction", x=0.5, y=0.5)
            return fig
        
        # Monthly aggregation
        monthly_data = expense_df.groupby(expense_df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
        monthly_data['Date_str'] = monthly_data['Date'].astype(str)
        monthly_data['Month_Num'] = range(len(monthly_data))
        
        # Simple linear regression for prediction
        from sklearn.linear_model import LinearRegression
        
        X = monthly_data[['Month_Num']]
        y = monthly_data['Amount']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next 3 months
        future_months = len(monthly_data) + np.arange(1, 4)
        future_predictions = model.predict(future_months.reshape(-1, 1))
        
        # Create future date labels
        last_date = monthly_data['Date'].iloc[-1]
        future_dates = []
        for i in range(1, 4):
            future_date = last_date + i
            future_dates.append(str(future_date))
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=monthly_data['Date_str'],
            y=monthly_data['Amount'],
            mode='lines+markers',
            name='Historical Expenses',
            line=dict(color=self.colors['expense'], width=3),
            marker=dict(size=8)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode='lines+markers',
            name='Predicted Expenses',
            line=dict(color='orange', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        fig.update_layout(
            title="Expense Prediction (Next 3 Months)",
            xaxis_title="Month",
            yaxis_title="Amount (₹)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_savings_target_chart(self, df):
        """Create savings target and progress visualization"""
        # Calculate current metrics
        total_income = df[df['Income/Expense'] == 'Income']['Amount'].sum()
        total_expense = df[df['Income/Expense'] == 'Expense']['Amount'].sum()
        current_savings = total_income - total_expense
        current_savings_rate = (current_savings / total_income * 100) if total_income > 0 else 0
        
        # Target savings rate (20%)
        target_savings_rate = 20
        target_savings = total_income * (target_savings_rate / 100)
        gap = target_savings - current_savings
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_savings_rate,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Savings Rate Progress"},
            delta = {'reference': target_savings_rate, 'suffix': "%"},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': "lightgreen" if current_savings_rate >= target_savings_rate else "orange"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgray"},
                    {'range': [10, 20], 'color': "gray"},
                    {'range': [20, 50], 'color': "lightgreen"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': target_savings_rate
                }
            }
        ))
        
        fig.update_layout(height=400, title="Savings Rate vs Target (20%)")
        
        return fig
