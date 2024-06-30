from typing import List, Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from ...messages.marketdata import MarketDataMsg, L2SubReqMsg
from ...messages.query import QuerySpreadResponseMsg
from ...orders import Side
from ..trading_agent import TradingAgent


class SpeculativeAgent(TradingAgent):
    """
    This agent compares the past max return with a threshold
    """

    def __init__(
        self,
        id: int,
        symbol,
        starting_cash,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        min_size=20,
        max_size=50,
        wake_up_freq: NanosecondTime = str_to_ns("60s"),
        poisson_arrival=True,
        order_size_model=None,
        subscribe=False,
        log_orders=False,
        conditional_on_PnL_leverage: bool = False

    ) -> None:

        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.size = (
            self.random_state.randint(self.min_size, self.max_size)
            if order_size_model is None
            else None
        )
        self.order_size_model = order_size_model  # Probabilistic model for order size
        self.wake_up_freq = wake_up_freq
        self.poisson_arrival = poisson_arrival  # Whether to arrive as a Poisson process
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq

        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscription_requested = False
        self.mid_list: List[float] = []
        self.avg_20_list: List[float] = []
        self.avg_50_list: List[float] = []
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"
        self.starting_cash = starting_cash

        self.last_mark_to_market = self.mark_to_market(self.holdings)
        self.mark_to_market_list = []



        self.max_buy_trigger = self.random_state.exponential(0.01)
        self.take_profit_trigger = self.random_state.exponential(50)
        self.stop_loss_trigger = self.random_state.exponential(0.001)
        self.look_back_window  = self.random_state.poisson(50)
        self.current_time = 0
        self.conditional_on_PnL_leverage = conditional_on_PnL_leverage

        print("new random params, heterogeneity")
        print(f"self.max_buy_trigger {self.max_buy_trigger }")
        print(f"self.take_profit_trigger {self.take_profit_trigger}")
        print(f"self.stop_loss_trigger {self.stop_loss_trigger}")
        print(f"self.look_back_window {self.look_back_window}")

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime) -> None:
        """Agent wakeup is determined by self.wake_up_freq"""
        can_trade = super().wakeup(current_time)
        if self.subscribe and not self.subscription_requested:
            super().request_data_subscription(
                L2SubReqMsg(
                    symbol=self.symbol,
                    freq=int(10e9),
                    depth=1,
                )
            )
            self.subscription_requested = True
            self.state = "AWAITING_MARKET_DATA"
        elif can_trade and not self.subscribe:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """Momentum agent actions are determined after obtaining the best bid and ask in the LOB"""
        super().receive_message(current_time, sender_id, message)
        self.current_time = current_time
        if (
            not self.subscribe
            and self.state == "AWAITING_SPREAD"
            and isinstance(message, QuerySpreadResponseMsg)
        ):
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
            self.place_orders(bid, ask)
            self.set_wakeup(current_time + self.get_wake_frequency())
            self.state = "AWAITING_WAKEUP"
        elif (
            self.subscribe
            and self.state == "AWAITING_MARKET_DATA"
            and isinstance(message, MarketDataMsg)
        ):
            bids, asks = self.known_bids[self.symbol], self.known_asks[self.symbol]
            if bids and asks:
                self.place_orders(bids[0][0], asks[0][0])
            self.state = "AWAITING_MARKET_DATA"
            self.mark_to_market_list.append(self.mark_to_market(self.holdings))


    def place_orders(self, bid: int, ask: int) -> None:
        """Speculative Agent actions logic. He simply buys if the recent max return is above a threshold.
        The heterogeneity comes from the lookback window and the thresholds"""

        if bid and ask:
            self.mid_list.append((bid + ask) / 2)
            if len(self.mid_list) > self.look_back_window:
                self.max_list.append(
                    SpeculativeAgent.max_return(self.mid_list, n=self.look_back_window).round(2)
                )
            # if len(self.mid_list) > 50:
            #     self.avg_50_list.append(
            #         SpeculativeAgent.max_return(self.mid_list, n=50).round(2)
            #     )
            if len(self.max_list) > 0:
                if self.order_size_model is not None:
                    self.size = self.order_size_model.sample(
                        random_state=self.random_state
                    )


                if self.size > 0:


                    # Buy if the max return is above the threshold
                    # no shorting allowed for the speculative agent?
                    # np.min(self.holdings[self.symbol],self.size)
                    #elif self.avg_50_list[-1] >=1.:
                    # if self.mark_to_market(self.holdings) - self.holdings["CASH"] > 20_000 and self.get_holdings(self.symbol) > 0:

                    # Can they get into debt?
                    #if self.get_holdings(self.symbol) > 0 and (self.mark_to_market(self.holdings) > 3*self.starting_cash or self.mark_to_market(self.holdings) < 0.1*self.starting_cash):
                    

                    # assert(self.mark_to_market(self.holdings)>=0) #NC: They can use leverage

                    # SELLING
                    if self.get_holdings(self.symbol) > 0 and (self.mark_to_market(self.holdings) > self.take_profit_trigger*self.last_mark_to_market or self.mark_to_market(self.holdings) < self.stop_loss_trigger*self.starting_cash):

          
          
                        assert(self.get_holdings(self.symbol) > 0)
                        assert(bid>0)

                        self.place_market_order(
                            self.symbol,
                            # No short selling allowed
                            quantity=min(self.get_holdings(self.symbol),self.size),
                            side=Side.ASK,
                        )
                    # BUYING
                    # The agent buys is the recently observed max return is above a level and the stop loss has not been hit
                    # elif self.holdings["CASH"] >= self.size * ask and (self.avg_20_list[-1] >= .01 and self.mark_to_market(self.holdings) >= 0.1*self.starting_cash):
                    elif self.max_list[-1] >= self.max_buy_trigger:
              
              

                        assert(ask>0)

                        # They can only buy with their cash if performance is negative
                        if ask and self.conditional_on_PnL_leverage and (self.mark_to_market(self.holdings) < self.starting_cash):
                                self.size = min(self.size, int(self.holdings["CASH"]/ask) )

                        if self.size>0:
                            self.place_market_order(
                                self.symbol,
                                quantity=self.size,
                                side=Side.BID,
                            )

                        self.last_mark_to_market =  self.mark_to_market(self.holdings)

        

    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)

            return int(round(delta_time))

    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    @staticmethod
    def max_return(a,n=20):
        # first we compute the daily log returns and then the max based on the last n mid values 
        max_found = np.max(np.diff(np.log(a[-n:])))
        assert(max_found>0)


        return max_found
