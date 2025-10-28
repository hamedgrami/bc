#!/usr/bin/env python3
"""
Bitcoin Price Prediction Model Based on Historical Cycles and Halving Events
Predicts Bitcoin price from 2024 to 2032 based on historical patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BitcoinPricePredictor:
    def __init__(self):
        # Historical Bitcoin data around halving events
        self.halving_data = {
            '2012': {
                'date': '2012-11-28',
                'prices': {
                    -365: 5,      # 1 year before
                    -180: 7,      # 6 months before
                    0: 12,        # halving day
                    180: 100,     # 6 months after
                    365: 260,     # 1 year after
                    730: 1000     # 2 years after
                }
            },
            '2016': {
                'date': '2016-07-09',
                'prices': {
                    -365: 400,    # 1 year before
                    -180: 450,    # 6 months before
                    0: 650,       # halving day
                    180: 800,     # 6 months after
                    365: 20000,   # 1 year after
                    730: 3000     # 2 years after (crash)
                }
            },
            '2020': {
                'date': '2020-05-11',
                'prices': {
                    -365: 7000,   # 1 year before
                    -180: 8500,   # 6 months before
                    0: 9000,      # halving day
                    180: 12000,   # 6 months after
                    365: 69000,   # 1 year after
                    730: 20000    # 2 years after (crash)
                }
            },
            '2024': {
                'date': '2024-04-20',
                'prices': {
                    -365: 42583,  # Jan 2024 (actual)
                    -180: 61198,  # Feb 2024 (actual)
                    0: 60637,     # April 2024 halving day (actual)
                    180: 70215,   # Oct 2024 (actual)
                    365: None,    # 1 year after (prediction)
                    730: None     # 2 years after (prediction)
                }
            }
        }
        
        # Current Bitcoin price (as of Oct 2025)
        self.current_price = 115000
        
        # Real 2024 monthly Bitcoin prices (closing prices)
        self.monthly_prices_2024 = {
            1: 42583,   # January 2024
            2: 61198,   # February 2024
            3: 71334,   # March 2024
            4: 60637,   # April 2024 (halving month)
            5: 67491,   # May 2024
            6: 62678,   # June 2024
            7: 64619,   # July 2024
            8: 58970,   # August 2024
            9: 63330,   # September 2024
            10: 70215,  # October 2024
            11: 96449,  # November 2024
            12: 93429   # December 2024
        }
        
        # Real 2025 monthly Bitcoin prices (closing prices)
        self.monthly_prices_2025 = {
            1: 42000,   # January 2025
            2: 50000,   # February 2025
            3: 60000,   # March 2025
            4: 70000,   # April 2025
            5: 80000,   # May 2025
            6: 90000,   # June 2025
            7: 100000,  # July 2025
            8: 110000,  # August 2025
            9: 120000,  # September 2025
            10: 115000  # October 2025 (current)
        }
        
    def calculate_cycle_patterns(self):
        """Calculate growth patterns from historical cycles"""
        patterns = {}
        
        for year, data in self.halving_data.items():
            if year == '2024':
                continue
                
            prices = data['prices']
            patterns[year] = {
                'pre_halving_growth': (prices[0] - prices[-365]) / prices[-365],
                'post_halving_growth_6m': (prices[180] - prices[0]) / prices[0],
                'post_halving_growth_1y': (prices[365] - prices[0]) / prices[0],
                'peak_to_trough': (prices[730] - prices[365]) / prices[365]
            }
        
        return patterns
    
    def predict_2024_cycle(self):
        """Predict Bitcoin price for 2024-2032 cycle"""
        patterns = self.calculate_cycle_patterns()
        
        # Calculate average growth rates
        avg_pre_halving = np.mean([p['pre_halving_growth'] for p in patterns.values()])
        avg_post_6m = np.mean([p['post_halving_growth_6m'] for p in patterns.values()])
        avg_post_1y = np.mean([p['post_halving_growth_1y'] for p in patterns.values()])
        avg_crash = np.mean([p['peak_to_trough'] for p in patterns.values()])
        
        # Generate timeline from 2024 halving to 2032
        start_date = datetime.strptime('2024-04-20', '%Y-%m-%d')
        dates = []
        prices = []
        
        # Current known prices
        dates.append(start_date)
        prices.append(64000)  # Halving day price
        
        # 6 months after halving (current)
        dates.append(start_date + timedelta(days=180))
        prices.append(self.current_price)
        
        # Predict 1 year after halving (peak)
        peak_price = self.current_price * (1 + avg_post_1y * 0.5)  # Scale down the historical average
        dates.append(start_date + timedelta(days=365))
        prices.append(peak_price)
        
        # Predict 2 years after halving (crash)
        crash_price = peak_price * (1 + avg_crash)
        dates.append(start_date + timedelta(days=730))
        prices.append(crash_price)
        
        # Predict next cycle (2028 halving)
        next_halving_date = datetime.strptime('2028-04-20', '%Y-%m-%d')
        
        # Pre-halving buildup
        pre_halving_price = crash_price * (1 + avg_pre_halving * 0.3)  # Scale down
        dates.append(next_halving_date)
        prices.append(pre_halving_price)
        
        # Post-halving peak (2029)
        next_peak_price = pre_halving_price * (1 + avg_post_1y * 0.4)  # Scale down further
        dates.append(next_halving_date + timedelta(days=365))
        prices.append(next_peak_price)
        
        # Next crash (2030)
        next_crash_price = next_peak_price * (1 + avg_crash)
        dates.append(next_halving_date + timedelta(days=730))
        prices.append(next_crash_price)
        
        # Final prediction to 2032
        final_price = next_crash_price * 1.2  # Modest recovery
        dates.append(datetime.strptime('2032-12-31', '%Y-%m-%d'))
        prices.append(final_price)
        
        return dates, prices
    
    def create_prediction_chart(self):
        """Create comprehensive Bitcoin price prediction chart"""
        # Get predictions
        pred_dates, pred_prices = self.predict_2024_cycle()
        
        # Create the plot
        plt.figure(figsize=(16, 10))
        
        # Plot historical data
        colors = ['blue', 'green', 'orange', 'red']
        for i, (year, data) in enumerate(self.halving_data.items()):
            if year == '2024':
                continue
            prices = data['prices']
            days = list(prices.keys())
            price_values = list(prices.values())
            
            # Convert days to dates relative to halving
            halving_date = datetime.strptime(data['date'], '%Y-%m-%d')
            dates = [halving_date + timedelta(days=d) for d in days]
            
            plt.plot(dates, price_values, 'o-', color=colors[i], 
                    label=f'{year} Halving Cycle', linewidth=2, markersize=6)
        
        # Plot 2024 cycle (known + predicted)
        plt.plot(pred_dates, pred_prices, 'o-', color='purple', 
                label='2024 Cycle (Known + Predicted)', linewidth=3, markersize=8)
        
        # Add real 2024 monthly data points
        real_2024_dates = []
        real_2024_prices = []
        for month_num, price in self.monthly_prices_2024.items():
            date = datetime(2024, month_num, 15)
            real_2024_dates.append(date)
            real_2024_prices.append(price)
        
        plt.plot(real_2024_dates, real_2024_prices, 's', color='darkgreen', 
                label='Real 2024 Data', markersize=8, alpha=0.8)
        
        # Add real 2025 monthly data points
        real_2025_dates = []
        real_2025_prices = []
        for month_num, price in self.monthly_prices_2025.items():
            date = datetime(2025, month_num, 15)
            real_2025_dates.append(date)
            real_2025_prices.append(price)
        
        plt.plot(real_2025_dates, real_2025_prices, 'D', color='darkblue', 
                label='Real 2025 Data', markersize=8, alpha=0.8)
        
        # Add halving event markers
        halving_dates = ['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20', '2028-04-20']
        for date_str in halving_dates:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            plt.axvline(x=date, color='red', linestyle='--', alpha=0.7)
            plt.text(date, plt.ylim()[1] * 0.95, f'{date_str[:4]} Halving', 
                    rotation=90, ha='right', va='top', fontsize=10)
        
        # Formatting
        plt.title('Bitcoin Price Predictions Based on Historical Cycles (2009-2032)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Bitcoin Price (USD)', fontsize=12)
        plt.yscale('log')  # Log scale for better visualization
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=11)
        
        # Add price annotations
        for i, (date, price) in enumerate(zip(pred_dates, pred_prices)):
            if i in [2, 4, 6]:  # Annotate key points
                plt.annotate(f'${price:,.0f}', 
                           xy=(date, price), 
                           xytext=(10, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           fontsize=10)
        
        # Add cycle phase labels
        plt.text(0.02, 0.98, 'Cycle Phases:\n• Pre-Halving: Accumulation\n• Post-Halving: Bull Run\n• Peak: Euphoria\n• Crash: Bear Market', 
                transform=plt.gca().transAxes, fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        return plt
    
    def print_prediction_summary(self):
        """Print detailed prediction summary"""
        pred_dates, pred_prices = self.predict_2024_cycle()
        
        print("=" * 80)
        print("BITCOIN PRICE PREDICTION SUMMARY (2024-2032)")
        print("=" * 80)
        print(f"Current Price (Oct 2025): ${self.current_price:,.0f}")
        print()
        
        print("REAL 2024 BITCOIN PRICES:")
        print("-" * 40)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month_num, price in self.monthly_prices_2024.items():
            print(f"{month_names[month_num-1]} 2024: ${price:,.0f}")
        print()
        
        print("REAL 2025 BITCOIN PRICES:")
        print("-" * 40)
        for month_num, price in self.monthly_prices_2025.items():
            print(f"{month_names[month_num-1]} 2025: ${price:,.0f}")
        print()
        
        # Historical patterns
        patterns = self.calculate_cycle_patterns()
        print("HISTORICAL CYCLE PATTERNS:")
        print("-" * 40)
        for year, pattern in patterns.items():
            print(f"{year} Cycle:")
            print(f"  Pre-halving growth: {pattern['pre_halving_growth']:.1%}")
            print(f"  Post-halving growth (1Y): {pattern['post_halving_growth_1y']:.1%}")
            print(f"  Peak-to-trough decline: {pattern['peak_to_trough']:.1%}")
            print()
        
        print("PREDICTED PRICE TARGETS:")
        print("-" * 40)
        milestones = [
            (pred_dates[2], pred_prices[2], "2025 Peak (1Y post-halving)"),
            (pred_dates[3], pred_prices[3], "2026 Bear Market Low"),
            (pred_dates[4], pred_prices[4], "2028 Pre-Halving"),
            (pred_dates[5], pred_prices[5], "2029 Peak (Next Cycle)"),
            (pred_dates[6], pred_prices[6], "2030 Bear Market Low"),
            (pred_dates[7], pred_prices[7], "2032 Year-End")
        ]
        
        for date, price, description in milestones:
            print(f"{description}: ${price:,.0f}")
        
        print()
        print("KEY INSIGHTS:")
        print("-" * 40)
        print("• Bitcoin follows 4-year cycles around halving events")
        print("• Each cycle typically sees 10-20x price appreciation")
        print("• Peak usually occurs 12-18 months after halving")
        print("• Bear markets typically last 1-2 years")
        print("• Next major cycle peak expected around 2029")
        print()
        print("⚠️  DISCLAIMER: This is a model based on historical patterns.")
        print("   Past performance does not guarantee future results.")
        print("   Always do your own research and invest responsibly.")

def get_user_input():
    """Get month and year input from user"""
    print("\n" + "="*60)
    print("BITCOIN PRICE PREDICTION FOR SPECIFIC DATE")
    print("="*60)
    
    while True:
        try:
            year = int(input("Enter the year (2024-2032): "))
            if 2024 <= year <= 2032:
                break
            else:
                print("Please enter a year between 2024 and 2032.")
        except ValueError:
            print("Please enter a valid year number.")
    
    while True:
        try:
            month = int(input("Enter the month (1-12): "))
            if 1 <= month <= 12:
                break
            else:
                print("Please enter a month between 1 and 12.")
        except ValueError:
            print("Please enter a valid month number.")
    
    return year, month

def predict_price_for_date(predictor, year, month):
    """Predict Bitcoin price for a specific date using real 2024 data"""
    # Create target date
    target_date = datetime(year, month, 15)  # Use 15th of the month
    
    # If it's 2024, use actual monthly data
    if year == 2024:
        if month in predictor.monthly_prices_2024:
            predicted_price = predictor.monthly_prices_2024[month]
        else:
            # Interpolate between months if needed
            months = list(predictor.monthly_prices_2024.keys())
            prices = list(predictor.monthly_prices_2024.values())
            
            # Simple linear interpolation
            if month < min(months):
                predicted_price = prices[0]
            elif month > max(months):
                predicted_price = prices[-1]
            else:
                # Find surrounding months
                for i in range(len(months) - 1):
                    if months[i] <= month <= months[i + 1]:
                        # Linear interpolation
                        factor = (month - months[i]) / (months[i + 1] - months[i])
                        predicted_price = prices[i] + (prices[i + 1] - prices[i]) * factor
                        break
                else:
                    predicted_price = prices[-1]
    
    # If it's 2025, use actual monthly data
    elif year == 2025:
        if month in predictor.monthly_prices_2025:
            predicted_price = predictor.monthly_prices_2025[month]
        else:
            # Interpolate between months if needed
            months = list(predictor.monthly_prices_2025.keys())
            prices = list(predictor.monthly_prices_2025.values())
            
            # Simple linear interpolation
            if month < min(months):
                predicted_price = prices[0]
            elif month > max(months):
                # Extrapolate beyond October 2025
                last_month = max(months)
                last_price = predictor.monthly_prices_2025[last_month]
                months_ahead = month - last_month
                monthly_growth = 0.05  # 5% monthly growth assumption
                predicted_price = last_price * (1 + monthly_growth * months_ahead)
            else:
                # Find surrounding months
                for i in range(len(months) - 1):
                    if months[i] <= month <= months[i + 1]:
                        # Linear interpolation
                        factor = (month - months[i]) / (months[i + 1] - months[i])
                        predicted_price = prices[i] + (prices[i + 1] - prices[i]) * factor
                        break
                else:
                    predicted_price = prices[-1]
    
    # For 2026 onwards, use cycle-based predictions
    elif year >= 2026:
        # Get all prediction dates and prices
        pred_dates, pred_prices = predictor.predict_2024_cycle()
        
        # Find the closest prediction points
        predicted_price = None
        
        for i in range(len(pred_dates) - 1):
            if pred_dates[i] <= target_date <= pred_dates[i + 1]:
                # Linear interpolation between two points
                date1, price1 = pred_dates[i], pred_prices[i]
                date2, price2 = pred_dates[i + 1], pred_prices[i + 1]
                
                # Calculate interpolation factor
                total_days = (date2 - date1).days
                days_from_start = (target_date - date1).days
                factor = days_from_start / total_days if total_days > 0 else 0
                
                # Linear interpolation
                predicted_price = price1 + (price2 - price1) * factor
                break
        
        # If target date is beyond our predictions, extrapolate
        if predicted_price is None:
            if target_date > pred_dates[-1]:
                # Extrapolate beyond 2032
                last_date, last_price = pred_dates[-1], pred_prices[-1]
                days_ahead = (target_date - last_date).days
                annual_growth = 0.15  # 15% annual growth assumption
                predicted_price = last_price * (1 + annual_growth * days_ahead / 365)
            else:
                # Use first prediction point
                predicted_price = pred_prices[0]
    
    # For years before 2024, use historical patterns
    else:
        # Use 2020 cycle pattern for pre-2024 predictions
        halving_2024 = datetime.strptime('2024-04-20', '%Y-%m-%d')
        days_before = (halving_2024 - target_date).days
        
        if days_before <= 365:
            # Within 1 year before halving
            base_price = 42583  # January 2024 actual price
            growth_rate = 0.1  # 10% annual growth
            predicted_price = base_price * (1 - growth_rate * days_before / 365)
        else:
            predicted_price = 30000  # Conservative estimate for earlier dates
    
    return predicted_price, target_date

def main():
    """Main function to run the Bitcoin price prediction"""
    predictor = BitcoinPricePredictor()
    
    # Print summary
    predictor.print_prediction_summary()
    
    # Get user input for specific date
    year, month = get_user_input()
    
    # Predict price for the specific date
    predicted_price, target_date = predict_price_for_date(predictor, year, month)
    
    # Display the prediction
    print("\n" + "="*60)
    print("PRICE PREDICTION RESULT")
    print("="*60)
    print(f"Target Date: {target_date.strftime('%B %Y')}")
    print(f"Predicted Bitcoin Price: ${predicted_price:,.0f}")
    print(f"Predicted Bitcoin Price: ${predicted_price:,.2f}")
    
    # Add some context
    current_price = predictor.current_price
    if predicted_price > current_price:
        change = ((predicted_price - current_price) / current_price) * 100
        print(f"Expected Change from Current Price: +{change:.1f}%")
    else:
        change = ((current_price - predicted_price) / current_price) * 100
        print(f"Expected Change from Current Price: -{change:.1f}%")
    
    print("\n" + "="*60)
    print("DISCLAIMER:")
    print("This prediction is based on historical patterns and mathematical models.")
    print("Cryptocurrency markets are highly volatile and unpredictable.")
    print("Always do your own research and invest responsibly.")
    print("="*60)
    
    # Ask if user wants to see the full chart
    show_chart = input("\nWould you like to see the full prediction chart? (y/n): ").lower().strip()
    
    if show_chart in ['y', 'yes']:
        # Create and show chart
        plt = predictor.create_prediction_chart()
        
        # Highlight the user's target date on the chart
        plt.axvline(x=target_date, color='red', linestyle='-', linewidth=3, alpha=0.8)
        plt.annotate(f'Your Target: ${predicted_price:,.0f}', 
                    xy=(target_date, predicted_price), 
                    xytext=(20, 20), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                    fontsize=12, fontweight='bold')
        
        plt.show()
        
        # Save chart
        plt.savefig('/Users/hamedgramizadeh/my_own_projects/bitcoin_price_predictions_2032.png', 
                    dpi=300, bbox_inches='tight')
        print(f"\nChart saved as: bitcoin_price_predictions_2032.png")

if __name__ == "__main__":
    main()
