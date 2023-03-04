# Designing an energy arbitrage strategy with linear programmin
## Task:
I wonder whether it is (theoretically) possible to generate a profit with price arbitrage by using an electric car as energy storage. The goal is to make as much money as possible from price arbitrage. Maximise profits.
## Technical inputs/boundaries:
• Simulate process of buying and selling electricity over a whole year, start at January 01 00:00 with a full battery

- Maximise profit over the whole year 

• Maximum storage (battery_max) is 50 kWh

• It is possible to either charge or discharge the battery with a maximum of 10 kWh per hour
(charge_capacity), take the price at the beginning of the hour to calculate profits (e.g. battery is charged with 30 kWh and price at 6pm is 0.30 cents/kwH, you can either buy 10 kWh for 3€ and have 40 kWh or sell 10 kWh for 3€ and have 20 kWh)

- Note that there are hours with negative electricity costs, so you can actually gain money while charging the car

• Charging and discharging is only possible at night between 6pm and 8am, during the day no charging is possible and the car loses 10 kWh state of charge due to driving without compensation
- ### SOCat6pm=SOCat8am–10kWh

• Minimum of 40 kWh battery charge (battery_morning) is required in the morning at 8am,
other than that there are no boundaries (full discharge during the night is possible as long as
it’s charged back to at least 40 kWh at 8am)

• Profits are accumulated over the whole year, prices are known in advance for any given hour
(hypothetical scenario)

# Problem and approach

The NYISO makes the next day's hourly energy prices available at 11am each day (NYISO 2019, references available at bottom of page). The battery system we design here will schedule the next 24 hours of battery operation (noon of the current day through the 11am hour of the next day) using this information. We will assume that after the daily price announcement from the NYISO, the next 36 hours of price information are available: the prices for the remaining 12 hours of the current day, as well as 24 hours worth of prices for the next day. Therefore the optimization time horizon will be 36 hours, to take advantage of all available future price information.

Since the operational schedule will be repeated each day, the last 12 hours of the 36 hour strategy will always be ignored. This is because new price information will become available at 11am the following day, which we can take advantage of. However these "extra 12 hours" of data are not wasted; I determined in initial experiments that having a 36 hour time horizon creates a more profitable arbitrage strategy than shorter horizons. This makes intuitive sense, because the more future information we can incorporate into the arbitrage strategy, the more profitable it should be. You can experiment with different horizons using the code below, although the horizon is assumed to be at least 24 hours here. The plan can be visualized as follows:

e battery is said to be a price taker, meaning its activities do not affect the price of energy. The price paid for power to charge the battery, and revenue from discharging, is the location based marginal price (LBMP), which takes in to account the system marginal price, congestion component, and marginal loss component (PJM Interconnection LLC). The goal is to maximize profit, given the day-ahead prices and the battery system's parameters.

In this scenario, where future prices are known and the battery system is a price taker, the problem of designing an operational strategy can be solved by linear programming (Salles et al. 2017, Sioshansi et al. 2009, Wang and Zhang 2018). In brief summary, linear programming is a well-known technique for either maximizing or minimizing some objective. In this case, we want to maximize profit. As long as the mathematical function describing the objective, known as the objective function, as well as the constraints of the system, can all be described as linear combinations of the decision variables, which define the operational strategy, linear programming can be used to optimize the system.

# Conclusions
We found that an energy arbitrage strategy for a grid-connected battery can be formulated using linear programming, assuming future prices are known over some time horizon. We showed that when operating under an illustrative set of system parameters and using real-world energy price data, such a system can generate an annual profit of $963.

Further optimization for increased profit may be possible, if prices are able to be accurately predicted beyond the 36 hour optimization horizon used here. The NYISO price determination involves a load forecasting model, that depends on economic and weather factors. It may be possible to include such factors in a price forecasting model to estimate future day-ahead market prices that are not yet public. In another interesting direction, Wang and Zhang (2018) show that reinforcement learning using historical price data can lead to higher profits than maximizing instantaneous profit, suggesting other possible approaches to maximizing profit from energy arbitrage.

I hope you found this post helpful for understanding how linear programming can be used to formulate an optimal arbitrage strategy if future prices are known.

# References

Sioshansi, Ramteen, et al. 2009. Estimating the Value of Electricity Storage in PJM: Arbitrage and Some Welfare Effects. Energy Economics 31:2, 269-277.

Wang, Hao and Zhang, Baosen, 2018. Energy Storage Arbitrage in Real-Time Markets via Reinforcement Learning. IEEE PES General Meeting.