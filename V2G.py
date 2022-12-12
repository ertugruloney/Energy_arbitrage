#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Basic settings and package import

get_ipython().system('pip install pulp')
import datetime
import pulp
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import time
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters


plt.style.use('seaborn-bright')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Import price data and convert to better format

data21 = pd.read_csv('2021.csv')
data21 = data21['MTU (CET/CEST),"Day-ahead Price [EUR/MWh]","Currency","BZN|DE-LU"'].str.split(',', expand=True)
data21[['start_date','end_date']] = data21[0].str.split('-', expand=True)
del data21[0]
del data21['end_date']
data21.rename(columns = {1:'price', 2:'currency'}, inplace = True)
data21['price']=data21['price'].str.replace('"', '')
data21['currency']=data21['currency'].str.replace('"', '')
data21['price'] = data21['price'].replace(r'^\s*$', np.nan, regex=True)
data21['price'] = data21['price'].astype(float)
data21['start_date'] = pd.to_datetime(data21['start_date'])
data21['start_time']=data21['start_date'].dt.strftime("%H:%M")
data21['start_date']=data21['start_date'].dt.date
print(data21)


# In[3]:


#Average price in kWh

mean_price = data21['price'].mean()
mean_price / 1000


# In[4]:


#Calculate average price per hour over the year

mean_price21 = pd.DataFrame(data21.groupby('start_time')['price'].mean())
print(mean_price21)


# In[5]:


#Plot mean prices (see comments in V2G task file)
mean_price21_indexed = mean_price21.reset_index()
mean_price21_indexed['start_time'] = mean_price21_indexed['start_time'].astype(str)

plt.figure(figsize=(15,5))
plt.xlabel("Uhrzeit", size="x-large")
plt.ylabel("Durchschnittspreis in €/MWh", size="x-large")
plt.grid(True)
#plt.title("Durchschnittliche Strompreise pro Stunde in 2021")
plt.plot(mean_price21_indexed['start_time'], mean_price21_indexed['price'], linewidth=3)


# In[6]:


#Analyse mean prices per month for further analysis (see comments in V2G task file)
data21_monthly = data21.copy(deep=True)

#Step 1: Convert the start_date column to datetime format
data21_monthly['start_date'] = pd.to_datetime(data21_monthly['start_date'])

#Step 2: Keep only teh
data21_monthly['start_date'] = data21_monthly['start_date'].dt.to_period('M')

data21_monthly_meaned = data21_monthly.groupby(by='start_date')['price'].mean()
data21_monthly_meaned_indexed = data21_monthly_meaned.reset_index()
print(data21_monthly_meaned_indexed)


# In[7]:


#Plot average monthly prices (incomplete: improve graph quality)

data21_monthly_meaned_indexed['start_date'] = data21_monthly_meaned_indexed['start_date'].astype(str)
ax = data21_monthly_meaned_indexed.plot(x ="start_date" , y="price" , kind="line" ,figsize=[15, 5], linewidth=0.7, alpha=0.6, color="#0000FF")
plt.show()


# ## Profit maximization

# ### Prepare data

# In[8]:


# Constants to filter price data
morning = 7     # time at which driver leaves the house (note: driver leaves house at the end of this hour, e.g. 7 means he is leaving at 08:00)
evening = 18    # time at which driver returns home

# Filter price data frame by hours in which charging / discharging is possible

def select_timestamp(time):
    hours = int(time.split(':')[0])
    return True if hours <= morning or hours >= evening else False 

data21['selection'] = data21['start_time'].apply(select_timestamp)
data21 = data21[data21['selection']]
data21


# In[9]:


def rename_timestamp(time):
    hours = int(time.split(':')[0])
    if hours >= 18:
        hours -= 10
    return f"{hours}:00"

data21['start_time_new'] = data21['start_time'].apply(rename_timestamp)
data21


# In[10]:


# Create time stamp for tracking later

data21['Time Stamp'] = data21['start_date'].astype(str) + " " + data21['start_time_new'].astype(str)
data21 = data21.set_index(['Time Stamp'])
data21.head(30)


# In[11]:


data21.index = pd.to_datetime(data21.index, format='%Y-%m-%d %H:%M')
data21


# In[12]:


data21 = data21.loc[:,['price']]
data21 = data21[~data21.index.duplicated(keep='first')]
data21['price'].fillna((data21['price'].mean()), inplace=True)
data21


# ### Build optimization model

# In[13]:


class Battery():
    def __init__(self,
                 time_horizon,
                 charge_capacity):
        #Set up decision variables for optimization.
        #These are the hourly charge and discharge flows for
        #the optimization horizon, with their limitations.
        self.time_horizon = time_horizon
    
        self.charge =         pulp.LpVariable.dicts(
            "charging_power",
            ('c_t_' + str(i) for i in range(0,self.time_horizon)),
            lowBound=0, upBound=charge_capacity,
            cat='Continuous')

        self.discharge =         pulp.LpVariable.dicts(
            "discharging_power",
            ('d_t_' + str(i) for i in range(0,self.time_horizon)),
            lowBound=0, upBound=charge_capacity,
            cat='Continuous')

    def set_objective(self, prices, efficiency):
        #Create a model and objective function.
        #This uses price data, which must have one price
        #for each point in the time horizon.
        try:
            assert len(prices) == self.time_horizon
        except:
            print('Error: need one price for each hour in time horizon')
        
        #Instantiate linear programming model to maximize the objective
        self.model = pulp.LpProblem("Energy arbitrage", pulp.LpMaximize)
    
        #Objective is profit
        #This formula gives the daily profit from charging/discharging
        #activities. Charging is a cost, discharging is a revenue
        
        #Edit for efficiency
        
        self.model +=         pulp.LpAffineExpression(
            [(self.charge['c_t_' + str(i)],
              -1*prices[i]) for i in range(0,self.time_horizon)]) +\
        pulp.LpAffineExpression(
            [(self.discharge['d_t_' + str(i)],
              prices[i]/efficiency) for i in range(0,self.time_horizon)])
        
    def add_storage_constraints(self,
                                battery_min,
                                battery_max,
                                battery_morning,
                                initial_level):
        #Storage level constraint 1
        #This says the battery cannot have less than zero energy, at
        #any hour in the horizon
     
        for hour_of_sim in range(1,self.time_horizon+1):     
            self.model +=             initial_level             + pulp.LpAffineExpression(
                [(self.charge['c_t_' + str(i)],1)
                 for i in range(0,hour_of_sim)]) \
            - pulp.lpSum(
                self.discharge[index]
                for index in('d_t_' + str(i)
                             for i in range(0,hour_of_sim)))\
            >= battery_min
            
        #Storage level constraint 2
        #Similar to 1
        #This says the battery cannot have more than the
        #battery capacity
        for hour_of_sim in range(1,self.time_horizon+1):
            self.model +=             initial_level             + pulp.LpAffineExpression(
                [(self.charge['c_t_' + str(i)], 1)
                 for i in range(0,hour_of_sim)]) \
            - pulp.lpSum(
                self.discharge[index]
                for index in ('d_t_' + str(i)
                              for i in range(0,hour_of_sim)))\
            <= battery_max
         
        #Storage level constraint 3
        #Battery should have at least minimum required state of energy in the morning
        for hour_of_sim in range(7,8):     
            self.model +=             initial_level             + pulp.LpAffineExpression(
                [(self.charge['c_t_' + str(i)],1)
                 for i in range(0,hour_of_sim)]) \
            - pulp.lpSum(
                self.discharge[index]
                for index in('d_t_' + str(i)
                             for i in range(0,hour_of_sim)))\
            >= battery_morning

        self.model += pulp.LpConstraint(self.charge['c_t_8'],sense=0,rhs=0)
        self.model += pulp.LpConstraint(self.discharge['d_t_8'],sense=0,rhs=10)

        
    def solve_model(self):
        #Solve the optimization problem
        self.model.solve()
        
        #Show a warning if an optimal solution was not found
        if pulp.LpStatus[self.model.status] != 'Optimal':
            print('Warning: ' + pulp.LpStatus[self.model.status])
            
    def collect_output(self,intiallevel):  
        #Collect hourly charging and discharging rates within the
        #time horizon
        hourly_charges =            np.array(
                [self.charge[index].varValue for
                 index in ('c_t_' + str(i) for i in range(0,time_horizon))])
        hourly_discharges =            np.array(
                [self.discharge[index].varValue for
                 index in ('d_t_' + str(i) for i in range(0,time_horizon))])
        
        #Add automatic discharge to simulate driving
        #Task: Insert a variable that can be changed
        lostenegry=[]
        currentengery=intiallevel
        for i in range(0,time_horizon):
            currentengeryN=currentengery + hourly_charges[i] - hourly_discharges[i]
            if currentengeryN>=currentengery:
                lostenegry.append(0)
            else:
                
                lostenegry.append(currentengery-currentengeryN)
            currentengery=currentengeryN
                
        
        hourly_discharges_updated = []
        for i in range(0,time_horizon):
            if i != morning:
                hourly_discharges_updated.append(self.discharge['d_t_' + str(i)].varValue)
            else:
                hourly_discharges_updated.append(0)

        return hourly_charges, hourly_discharges, hourly_discharges_updated,lostenegry


# In[14]:


def simulate_battery(initial_level,
                     price_data,
                     efficiency,
                     charge_capacity,
                     battery_max,
                     time_horizon,
                     start_day):
    #Track simulation time (optional)
    #tic = time.time()
    
    #Initialize output variables
    all_hourly_lostenegery = np.empty(0)
    all_hourly_charges = np.empty(0)
    all_hourly_discharges = np.empty(0)
    all_hourly_state_of_energy = np.empty(0)
    all_daily_discharge_throughput = np.empty(0)
    
    #Set up decision variables for optimization by
    #instantiating the Battery class
    battery = Battery(
        time_horizon=time_horizon,
        charge_capacity=charge_capacity)
    
    #############################################
    #Run the optimization for each day of the year.
    #############################################
    
    #There are 365 24-hour periods (noon to noon) in the simulation,
    #contained within 365 days
    for day_count in range(365):
        #print('Trying day {}'.format(day_count))
        
        #############################################
        ### Select data and simulate daily operation
        #############################################
        
        #Set up the 36 hour optimization horizon for this day by
        #adding to the first day/time of the simulation
        start_time = start_day         + pd.Timedelta(day_count, unit='days')
        end_time = start_time + pd.Timedelta(time_horizon-1, unit='hours')
        #print(start_time, end_time)
    
        #Retrieve the price data that will be used to calculate the
        #objective
        prices =         price_data[start_time:end_time]['price'].values
                      
        #Create model and objective
        battery.set_objective(
            prices,
            efficiency=efficiency)

        #Set storage constraints
        battery.add_storage_constraints(
            battery_min=battery_min,
            battery_max=battery_max,
            battery_morning=battery_morning,
            initial_level=initial_level)


        #Solve the optimization problem and collect output
        battery.solve_model()
        hourly_charges, hourly_discharges, hourly_discharges_updated,lostenegry = battery.collect_output(initial_level)
        
        #############################################
        ### Manipulate daily output for data analysis
        #############################################
        
        #Collect daily discharge throughput
        daily_discharge_throughput = sum(hourly_discharges)
        #Calculate net hourly power flow (kW), needed for state of energy.
        #Charging needs to factor in efficiency, as not all charged power
        #is available for discharge.
        net_hourly_activity = hourly_charges - hourly_discharges
        #Cumulative changes in energy over time (kWh) from some baseline
        cumulative_hourly_activity = np.cumsum(net_hourly_activity)
        #Add the baseline for hourly state of energy during the next
        #time step (t2)
        state_of_energy_from_t2 = initial_level + cumulative_hourly_activity
        
        #Append output
        all_hourly_charges = np.append(all_hourly_charges, hourly_charges)
        all_hourly_lostenegery = np.append(all_hourly_lostenegery, lostenegry)
        all_hourly_discharges = np.append(
            all_hourly_discharges, hourly_discharges_updated)
        all_hourly_state_of_energy =         np.append(all_hourly_state_of_energy, state_of_energy_from_t2)
        all_daily_discharge_throughput =         np.append(
            all_daily_discharge_throughput, daily_discharge_throughput)
        
        #############################################
        ### Set up the next day
        #############################################
        
        #Initial level for next period is the end point of current period
        initial_level = state_of_energy_from_t2[-1]
        
        

    #toc = time.time()
        
    #print('Total simulation time: ' + str(toc-tic) + ' seconds')

    return all_hourly_charges, all_hourly_discharges,         all_hourly_state_of_energy,        all_daily_discharge_throughput, all_hourly_lostenegery


# In[196]:


# Set constants for optimization model
time_horizon = 14
battery_max = 50              # maximum SOC at all times / car battery capacity 50
battery_min = 0               # minimum SOC at all times
battery_morning = 40         # minimum SOC at 8am
charge_capacity = 22          # capacity of charger, maximum kWh a car can be charged/discharged within an hour
efficiency = 0.85             # round trip efficiency of charging
initial_level = 50            # initial SOC at beginning of the year
#driving = 10                 # variable to adjust driving consumption
#add another constant to control energy lost due to driving


# In[197]:


all_hourly_charges, all_hourly_discharges, all_hourly_state_of_energy,all_daily_discharge_throughput ,all_hourly_lostenegery= simulate_battery(initial_level=initial_level,
                 price_data=data21,
                 charge_capacity=charge_capacity,
                 battery_max=battery_max,
                 efficiency=efficiency,
                 time_horizon=time_horizon,
                 start_day=pd.Timestamp(
                     year=2021, month=1, day=1, hour=0))


# ## Result analysis

# In[198]:


data_final = data21.copy()


# In[199]:


#Analyse profits

#These indicate flows during the hour of the datetime index

data_final['Charging power (kW)'] = all_hourly_charges
data_final['Discharging power (kW)'] = all_hourly_discharges
data_final['Power output (kW)'] =     all_hourly_discharges - all_hourly_charges
#This is the state of power at the beginning of the hour of the datetime index 
data_final['State of Energy (kWh)'] =     np.append(initial_level, all_hourly_state_of_energy[0:-1])


# In[200]:


data_final['Revenue generation ($)'] = data_final['Discharging power (kW)'] * data_final['price'] / 1000                #divide by 1000 to convert from MWh to kWh


# In[201]:


data_final['Charging cost ($)'] = data_final['Charging power (kW)'] * data_final['price'] / 1000                #divide by 1000 to convert from MWh to kWh


# In[202]:


data_final['Profit ($)'] = data_final['Revenue generation ($)'] - data_final['Charging cost ($)']


# In[203]:


data_final[:50]


# In[204]:


data_final_wrong = data_final[data_final['State of Energy (kWh)']<0]
data_final_wrong


# In[205]:


data_final_neutral = data_final[data_final['Power output (kW)']==0]
data_final_neutral


# In[206]:


data_final_charging = data_final[data_final['Power output (kW)']<0] #.sort_values('price')
data_final_charging


# In[207]:


data_final_discharging = data_final[data_final['Power output (kW)']>0]
data_final_discharging


# ### Profit

# In[208]:


discharging_revenue = (data_final_discharging['Power output (kW)'] * (data_final_discharging['price'] / 1000)).sum()
discharging_revenue


# In[209]:


charging_cost = (data_final_charging['Power output (kW)'] * (data_final_charging['price'] / 1000)).sum() * -1
charging_cost


# In[210]:


total_profit = discharging_revenue - charging_cost
total_profit


# In[211]:


#Calculate driving cost by taking the minimum price
charging_min = data_final_charging.resample('D')['price'].min()
driving_cost = charging_min.sum() * 10 / 1000
driving_cost


# In[212]:


profit_arbitrage = discharging_revenue - (charging_cost - driving_cost)
profit_arbitrage


# In[213]:


#average spot price in €/kWh when charging 
average_charging_price = data_final_charging['price'].mean() / 1000
average_charging_price


# In[214]:


#average spot price in €/kWh when discharging 
average_discharging_price = data_final_discharging['price'].mean() / 1000
average_discharging_price


# In[215]:


average_discharging_price - average_charging_price


# In[216]:


fig,ax=plt.subplots(2,1,figsize=(10,5), constrained_layout=True)
#ax.set_ylabel(labels, rotation=45)

monate=['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
y1=data_final['Profit ($)'].resample('M').sum().tolist()
#y2=monthly_discharging['Power output (kW)'].tolist()
y3=data21['price'].resample('H').sum()


#ax[1].bar(monate,y2,color="blue")
#ax[1].legend(["Monatliche Einspeisung (kW)"], loc='upper left')
ax[0].plot(data21['price'].resample('H').sum())
ax[0].legend(["Strompreis (€/MWh)"])
ax[0].set_ylabel("€/MWh", loc='top', rotation=0)
ax[0].xaxis.set_visible(False)
ax[1].bar(monate,y1,color="blue")
ax[1].legend(["Monatlicher Gewinn (€)"], loc='upper left')
ax[1].set_ylabel("€", loc='top', rotation=0)

#plt.savefig('Customed Plot.png', dpi=500, bbox_inches='tight')


# In[217]:


monthly_profit = data_final['Profit ($)'].resample('M').sum().tolist()
monthly_discharging = (data_final_discharging['price'].resample('M').mean() /1000).tolist() #/ 1000
monthly_charging = (data_final_charging['price'].resample('M').mean() / 1000).tolist()
monthly_average = (data21['price'].resample('M').mean() / 1000).tolist()
monthly_output = (data_final_discharging['Power output (kW)'].resample('M').sum()).tolist()
monthly_data = pd.DataFrame([monthly_average, monthly_discharging, monthly_charging, monthly_profit, monthly_output])
monthly_data = monthly_data.T
monthly_data.columns = ['Preis', 'Entladen', 'Laden', 'Profit', 'Einspeisung']
monthly_data['Monat'] = monate
monthly_data['Spread'] = round(monthly_data['Entladen'] - monthly_data['Laden'], 4)
monthly_data['Marge'] = round(monthly_data['Spread'] / monthly_data['Entladen'] * 100, 2)
monthly_data = monthly_data[['Monat', 'Preis','Entladen', 'Laden', 'Spread', 'Marge', 'Profit', 'Einspeisung']]
monthly_data['Profit'] = round(monthly_data['Profit'], 2)
monthly_data['Preis'] = round(monthly_data['Preis'], 4)
monthly_data['Entladen'] = round(monthly_data['Entladen'], 4)
monthly_data['Laden'] = round(monthly_data['Laden'], 4)
monthly_data = monthly_data.set_index("Monat")
monthly_data


# ### Batterienutzung

# In[218]:


#total charging power
data_final_charging['Power output (kW)'].sum() * -1


# In[219]:


#average charger capacity used when charging
data_final_charging['Power output (kW)'].mean() * -1


# In[220]:


#total discharging power output
data_final_discharging['Power output (kW)'].sum()


# In[190]:


#average charger capacity used when discharging
data_final_discharging['Power output (kW)'].mean()


# In[191]:


data_final_neutral_indexed = data_final_neutral.reset_index()
data_final_neutral_indexed['time'] = pd.to_datetime(data_final_neutral_indexed['Time Stamp']).dt.time
data_final_neutral_indexed['time'].value_counts()


# In[192]:


data_final_charging_indexed = data_final_charging.reset_index()
data_final_charging_indexed['time'] = pd.to_datetime(data_final_charging_indexed['Time Stamp']).dt.time
data_final_charging_indexed['time'].value_counts()


# In[193]:


data_final_discharging_indexed = data_final_discharging.reset_index()
data_final_discharging_indexed['time'] = pd.to_datetime(data_final_discharging_indexed['Time Stamp']).dt.time
data_final_discharging_indexed['time'].value_counts()


# In[194]:


time_analysis = pd.DataFrame([["18:00", 365, 0, 0], ["19:00", 354, 1, 10], ["20:00", 360, 0, 5], ["21:00", 242, 3, 120],
                             ["22:00", 120, 0, 245], ["23:00", 7, 1, 357], ["00:00", 1, 128, 236], ["01:00", 1, 270, 94],
                             ["02:00", 4, 307, 54], ["03:00", 1, 356, 8], ["04:00", 1, 351, 13], ["05:00", 2, 232, 131],
                             ["06:00", 86, 90, 189], ["07:00", 0, 141, 224]],
                             columns=['Uhrzeit', 'Entladen', 'Laden', 'Neutral'])

print(time_analysis)

time_analysis.plot(x='Uhrzeit', kind='bar', stacked=True, figsize=(12,8), fontsize='large', rot=0,
                  color={"Entladen":"red", "Laden":"green", "Neutral":"darkgray"})
plt.xlabel("Uhrzeit", fontsize='large')
plt.legend(fontsize='large', loc='upper right', fancybox=True, shadow=True)

plt.savefig('Batterie nach Uhrzeit.png', dpi=1000, bbox_inches='tight')


# In[195]:


monthly_discharging=data_final_discharging['Power output (kW)'].resample('M').sum().reset_index()
monthly_discharging


# In[46]:


xwerte=['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
ywerte=monthly_discharging['Power output (kW)'].tolist()
plt.bar(xwerte, ywerte)
plt.ylabel("Einspeisung (kWh)")
plt.figure(figsize=(16,10))
#plt.savefig('Einspeisung', dpi=1000, bbox_inches='tight')


# In[47]:


data_morning = data_final.at_time('07:00')
data_morning[:50]


# In[48]:


#Analyse final data
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.hist(all_hourly_state_of_energy)
plt.xlabel("kWh", fontsize="large")
plt.ylabel("Anzahl Stunden", fontsize="large")
plt.title('')

plt.subplot(2, 1, 2)
plt.hist(data_morning["State of Energy (kWh)"])
plt.xlabel('kWh', fontsize="large")
plt.ylabel("Anzahl Stunden", fontsize="large")
plt.title('')
#plt.savefig('Akkuladestand 100 kWh.png', dpi=1000, bbox_inches='tight')


# ## Ergebnisse Sensitivitätsanalyse

# In[49]:


#Gewinn nach Batteriegröße

battery_analysis = pd.DataFrame([[20, 206.65], [25, 291.92], [30, 375.80], [35, 449.98], [40, 518.45], [45, 580.01], 
                                 [50, 630.66], [55, 677.67], [60, 706.16], [65, 734.82], [70, 737.44], [75, 738.22],
                                 [80, 738.45],[85, 738.64], [90, 738.74], [95, 738.74], [100, 738.74]],
                               columns=['Batteriegröße', 'Gewinn'])

print(battery_analysis)

SI_battery = (battery_analysis['Gewinn'].max() - battery_analysis['Gewinn'].min()) / battery_analysis['Gewinn'].max()
print(SI_battery)

battery_analysis.plot(x='Batteriegröße', xticks=(battery_analysis['Batteriegröße']), kind='line', figsize=(12,6), 
                      fontsize='large', rot=0, legend=(), linewidth=3)
plt.xlabel("Batteriekapazität (kWh)", fontsize='large')
plt.ylabel("Gewinn (€)", fontsize='large')

#plt.savefig('Gewinn nach Batterie.png', dpi=1000, bbox_inches='tight')


# In[ ]:





# In[ ]:





# In[ ]:




