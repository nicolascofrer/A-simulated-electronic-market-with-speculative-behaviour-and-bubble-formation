# RMSC-4 (Reference Market Simulation Configuration):
# - 1     Exchange Agent
# - 2     Adaptive Market Maker Agents
# - 102   Value Agents
# - 12    Momentum Agents
# - 1000  Noise Agents

import os
from datetime import datetime

import numpy as np
import pandas as pd

from abides_core.utils import get_wake_time, str_to_ns
from abides_markets.agents import (
    ExchangeAgent,
    NoiseAgent,
    ValueAgent,
    AdaptiveMarketMakerAgent,
    MomentumAgent,
    SpeculativeAgent,
)
from abides_markets.models import OrderSizeModel
# from abides_markets.oracles import SparseMeanRevertingOracle
from abides_markets.oracles import MeanRevertingOracle


from abides_markets.utils import generate_latency_model


########################################################################################################################
############################################### GENERAL CONFIG #########################################################


def build_config(
    # seed=int(datetime.now().timestamp() * 1_000_000) % (2 ** 32 - 1),
    seed=2023,
    date="20210205",
    end_time="09:35:00",
    sim_time = 1000,
    stdout_log_level="INFO",
    ticker="ASSET",
    starting_cash=10_000_000,  # Cash in this simulator is always in CENTS.
    log_orders=True,  # if True log everything
    # 1) Exchange Agent
    book_logging=True,
    book_log_depth=10,
    stream_history_length=500,
    exchange_log_orders=None,
    # 2) Noise Agent
    num_noise_agents=0,
    # 3) Value Agents
    #num_value_agents=1000,

    num_value_agents=500,
    r_bar=10_000,  # true mean fundamental value
    kappa=1,  # Value Agents appraisal of mean-reversion
    news_reaction=1,
    #lambda_a=5.7e-12,  # ValueAgent arrival rate
    # oracle
    kappa_oracle=0.5,  # Mean-reversion of fundamental time series.
    sigma_s=10_000,
    mean_s=100,
  
    # 4) Market Maker Agents
    # each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
    mm_window_size="adaptive",
    mm_pov=0.025,
    mm_num_ticks=5,
    mm_wake_up_freq= 15,#"10S",
    mm_min_order_size=1,
    mm_skew_beta=0,
    mm_price_skew=100,
    mm_level_spacing=5,
    mm_spread_alpha=0.5,
    mm_backstop_quantity=10_000,
    mm_cancel_limit_delay=30,  # 50 nanoseconds
    # 5) Momentum Agents
    num_momentum_agents=0,
    # num_momentum_agents=100,

    num_speculative_agents=0,
    agents_wake_up_freq = 15 #str_to_ns("30s")

):
    """
    create the background configuration for rmsc04
    These are all the non-learning agent that will run in the simulation
    :param seed: seed of the experiment
    :type seed: int
    :param log_orders: debug mode to print more
    :return: all agents of the config
    :rtype: list
    """

    print(f"build version 01-06-2023 18:40 with seed {seed}")
    # fix seed
    np.random.seed(seed)

    # def path_wrapper(pomegranate_model_json):
    #     """
    #     temporary solution to manage calls from abides-gym or from the rest of the code base
    #     TODO:find more general solution
    #     :return:
    #     :rtype:
    #     """
    #     # get the  path of the file
    #     path = os.getcwd()
    #     if path.split("/")[-1] == "abides_gym":
    #         return "../" + pomegranate_model_json
    #     else:
    #         return pomegranate_model_json

    # mm_wake_up_freq = str_to_ns(mm_wake_up_freq)

    # order size model
    ORDER_SIZE_MODEL = OrderSizeModel()  # Order size model
    # market marker derived parameters
    MM_PARAMS = [
        (mm_window_size, mm_pov, mm_num_ticks, mm_wake_up_freq, mm_min_order_size)
       # (mm_window_size, mm_pov, mm_num_ticks, mm_wake_up_freq, mm_min_order_size),
    ]
    NUM_MM = len(MM_PARAMS)
    # noise derived parameters
    SIGMA_N = r_bar / 100  # observation noise variance

    # date&time
    DATE = int(pd.to_datetime(date).to_datetime64())

    print(f"DATE {DATE}")
    MKT_OPEN = DATE
    MKT_CLOSE = DATE + sim_time

    print(f"MKT_OPEN {MKT_OPEN}")
    print(f"MKT_CLOSE {MKT_CLOSE}")

    # These times needed for distribution of arrival times of Noise Agents


    # oracle
    oracle_seed = np.random.randint(low=0, high=2 ** 32)
    print(f"oracle seed {oracle_seed}")
    symbols = {
        ticker: {
            "r_bar": r_bar,
            "kappa": kappa_oracle,
            "sigma_s": sigma_s,
            "mean_s": mean_s
            # "fund_vol": fund_vol,
            # "megashock_lambda_a": megashock_lambda_a,
            # "megashock_mean": megashock_mean,
            # "megashock_var": megashock_var,
            # "random_state": np.random.RandomState(
            #     # seed=np.random.randint(low=0, high=2 ** 32)
            #     seed=oracle_seed
            # ),
        }
    }

    # oracle = SparseMeanRevertingOracle(MKT_OPEN, NOISE_MKT_CLOSE, symbols)
    oracle = MeanRevertingOracle(MKT_OPEN, MKT_CLOSE, symbols)

    # Agent configuration
    agent_count, agents, agent_types = 0, [], []

    agents.extend(
        [
            ExchangeAgent(
                id=0,
                name="EXCHANGE_AGENT",
                type="ExchangeAgent",
                mkt_open=MKT_OPEN,
                mkt_close=MKT_CLOSE,
                symbols=[ticker],
                book_logging=book_logging,
                book_log_depth=book_log_depth,
                log_orders=exchange_log_orders,
                pipeline_delay=0,
                computation_delay=0,
                stream_history=stream_history_length,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
                ),
            )
        ]
    )
    agent_types.extend("ExchangeAgent")
    agent_count += 1

    
    agents.extend(
        [
            ValueAgent(
                id=j,
                name="Value Agent {}".format(j),
                type="ValueAgent",
                symbol=ticker,
                starting_cash=starting_cash,
                sigma_n=SIGMA_N,
                r_bar=r_bar,
                kappa=kappa,
                news_reaction=news_reaction,
                #lambda_a=lambda_a,
                wake_up_freq=agents_wake_up_freq,
                log_orders=log_orders,
                order_size_model=ORDER_SIZE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
                ),
            )
            for j in range(agent_count, agent_count + num_value_agents)
        ]
    )
    agent_count += num_value_agents
    agent_types.extend(["ValueAgent"])

    
    ## Adding the new speculative agents
    agents.extend(
        [
            SpeculativeAgent(
                id=j,
                name="SPECULATIVE_AGENT_{}".format(j),
                type="SpeculativeAgent",
                symbol=ticker,
                starting_cash=starting_cash,
                min_size=1,
                max_size=10,
                wake_up_freq=agents_wake_up_freq,
                poisson_arrival=True,
                log_orders=log_orders,
                order_size_model=ORDER_SIZE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
                ),
            )
            for j in range(agent_count, agent_count + num_speculative_agents)
        ]
    )
    agent_count += num_speculative_agents
    agent_types.extend("SpeculativeAgent")
    # extract kernel seed here to reproduce the state of random generator in old version
    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
    )



    agents.extend(
        [
            AdaptiveMarketMakerAgent(
                id=j,
                name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                type="AdaptivePOVMarketMakerAgent",
                symbol=ticker,
                starting_cash=starting_cash,
                pov=MM_PARAMS[idx][1],
                min_order_size=MM_PARAMS[idx][4],
                window_size=MM_PARAMS[idx][0],
                num_ticks=MM_PARAMS[idx][2],
                wake_up_freq=MM_PARAMS[idx][3],
                poisson_arrival=True,
                cancel_limit_delay=mm_cancel_limit_delay,
                skew_beta=mm_skew_beta,
                price_skew_param=mm_price_skew,
                level_spacing=mm_level_spacing,
                spread_alpha=mm_spread_alpha,
                backstop_quantity=mm_backstop_quantity,
                log_orders=log_orders,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
                ),
            )
            for idx, j in enumerate(range(agent_count, agent_count + NUM_MM))
        ]
    )
    agent_count += NUM_MM
    agent_types.extend("POVMarketMakerAgent")


    
    # LATENCY
    latency_model = generate_latency_model(agent_count,latency_type="no_latency")

    default_computation_delay = 0  # 50 nanoseconds

    ##kernel args
    # kernelStartTime = DATE
    # kernelStopTime = MKT_CLOSE + str_to_ns("1s")
    kernelStartTime = MKT_OPEN
#    kernelStopTime = MKT_CLOSE + str_to_ns("1s")
    kernelStopTime = MKT_CLOSE
    return {
        "seed": seed,
        "start_time": kernelStartTime,
        "stop_time": kernelStopTime,
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": default_computation_delay,
        "custom_properties": {"oracle": oracle},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": stdout_log_level,
    }
