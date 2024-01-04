import json
import threading
import asyncio
import requests

import websockets
from datetime import datetime, timezone, timedelta

from .base import *
from common import logging_helper as logging


def utc_to_local(utc_dt):
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)


def aslocaltimestr(utc_dt):
    return utc_to_local(utc_dt).strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_transaction_date(datetime_str):
    formats = ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ']

    for fmt in formats:
        try:
            datetime_obj = datetime.strptime(datetime_str, fmt)
            return datetime_obj
        except ValueError:
            continue

    # Return None if no format matched
    return None


class TradierBroker(Broker):
    def __init__(self, base_url, ws_base_url, basic_token, account_id, bearer_token, log_order, simreq,
                 live_trading=False):
        self.api_token = basic_token
        self.base_url = base_url
        self.websocket_base_url = ws_base_url
        self.log_order = log_order
        self.sim_req = simreq

        self.basic_headers = {
            'Authorization': f'Basic {basic_token}',
            'Accept': 'application/json'
        }
        self.bearer_headers = {'Authorization': f'Bearer {bearer_token}', 'Accept': 'application/json'}
        self.account_id = account_id

        self.broker_nickname = 'TRD'
        self.live_trading = live_trading
        self.account_session_id = None
        self.positions = []
        self.opened_orders = []
        self.orders_tp_sl_map = {}

        if account_id == '':
            self.create_virtual_accounts()

        # open account session and connect websocket
        self._create_account_session()
        threading.Thread(target=self._run_streaming_order_event).start()

        # get data
        self.get_active_positions()
        self.get_orders()

    @staticmethod
    def _get_order_class_string(trading_order_type: TradingOrderType) -> str:
        if trading_order_type == TradingOrderType.Equity:
            return 'equity'
        elif trading_order_type == TradingOrderType.OTO:
            return 'oto'
        elif trading_order_type == TradingOrderType.OCO:
            return 'oco'
        else:
            return 'otoco'

    @staticmethod
    def _get_order_type_string(order_type: OrderType) -> str:
        if order_type == OrderType.Market:
            return 'market'
        elif order_type == OrderType.Limit:
            return 'limit'
        elif order_type == OrderType.Stop:
            return 'stop'
        else:
            return 'stop_limit'

    @staticmethod
    def _get_order_side_string(order_side: OrderSide) -> str:
        if order_side == OrderSide.Buy:
            return 'buy'
        elif order_side == OrderSide.Sell:
            return 'sell'
        elif order_side == OrderSide.SellShort:
            return 'sell_short'
        elif order_side == OrderSide.BuyToCover:
            return 'buy_to_cover'
        else:
            return 'none'

    def _get_main_orders(self, **kwargs) -> list:
        endpoint_url = f'/main_orders/?broker={self.broker_nickname}'
        for key, value in kwargs.items():
            endpoint_url += f'&{key}={value}'
        resp = self.sim_req.get_response(f'{self.rest_api_orders_url}{endpoint_url}', attach_slash=False)
        if resp is None:
            return []
        return resp.json()['results']

    def _save_new_order(self, params) -> dict:
        # Save new order to database via Rest API
        if not self.log_order:
            return {}

        open_order_date = parse_transaction_date(params['create_date'])
        if open_order_date is None:
            logging.error(f'Invalid create date returned: {params["create_date"]}')
            return {}
        open_order_date_est = get_est_time(open_order_date)

        try:
            # Order Type
            if params['type'] == 'market':
                order_type = 'MKT'
            elif params['type'] == 'limit':
                order_type = 'LMT'
            elif params['type'] == 'stop':
                order_type = 'STP'
            else:
                order_type = 'STPLMT'

            # Order Side
            if params['side'] == 'buy':
                order_side = 'BUY'
            elif params['side'] == 'sell':
                order_side = 'SELL'
            elif params['side'] == 'sell_short':
                order_side = 'SELL SHORT'
            else:
                order_side = 'BUY TO COVER'

            # Order Status
            order_status = 'OP'
            if params['status'] == 'filled':
                order_status = 'FL'
            elif params['status'] == 'canceled':
                order_status = 'CN'
            elif params['status'] == 'rejected':
                order_status = 'RJ'

            order_data = {
                'broker': self.broker_nickname,
                'order_id': params['id'],
                'client_id': self.account_id,
                'symbol': params['symbol'],
                'snapshot': int(params['tag'].split('-')[1]),
                'order_type': order_type,
                'order_side': order_side,
                'quantity': params['quantity'],
                'status': order_status,
                'period': 1,
                'open_time': open_order_date_est.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'live_trading': self.live_trading
            }
            if 'price' in params:
                order_data['take_profit_price'] = params['price']
            if 'stop_price' in params:
                order_data['stop_loss_price'] = params['stop_price']

            if params['tag'].split('-')[2] == '0':  # main order opened
                if str(params['id']) in self.orders_tp_sl_map:
                    order_meta_data = self.orders_tp_sl_map[str(params['id'])]
                    order_data['forecast'] = order_meta_data['forecast_id']
                    order_data['current_price'] = order_meta_data['current_price']
                    order_data['forecast_price'] = order_meta_data['forecast_price']
                    order_data['take_profit_price'] = order_meta_data['take_profit_price']
                    order_data['stop_loss_price'] = order_meta_data['stop_loss_price']
                    del self.orders_tp_sl_map[str(params['id'])]

            if int(params['tag'].split('-')[2]) != 0:
                order_data['parent_order_id'] = int(params['tag'].split('-')[2])

            new_order = self.sim_req.post_response(self.rest_api_orders_url, order_data)

            if new_order:
                logging.info(f'Order saved success: {new_order}')

            return new_order

        except Exception as e:
            logging.error(f'Save order failed: {e}')
            return {}

    def _open_tradier_order(self, params: dict):
        try:
            endpoint_url = f'/v1/accounts/{self.account_id}/orders'

            response = requests.post(
                self.base_url + endpoint_url,
                data=params,
                headers=self.bearer_headers
            )
            if response.status_code != 200:
                return None
            logging.info(f'Open order response: {response.text}')

            json_response = response.json()
            if json_response['order']['status'] != 'ok':
                return None

            return json_response['order']['id']

        except Exception as e:
            logging.error(f'Order cannot be placed, {params} {e}')
            return None

    def _cancel_tradier_order(self, order_id: int) -> bool:
        try:
            endpoint_url = f'/v1/accounts/{self.account_id}/orders/{order_id}'
            response = requests.delete(
                self.base_url + endpoint_url,
                headers=self.bearer_headers
            )
            if response.status_code >= 300:
                return False

        except Exception as e:
            logging.error(f'Cannot cancel order {order_id}, {e}')
            return False

        return True

    def _create_account_session(self) -> bool:
        endpoint_url = '/v1/accounts/events/session'
        response = requests.post(
            self.base_url + endpoint_url,
            data={},
            headers=self.bearer_headers
        )
        if response.status_code >= 300:
            return False

        json_response = response.json()
        try:
            account_session_id = json_response['stream']['sessionid']
        except Exception as e:
            logging.error(f'Failed to get session id: {e}')
            account_session_id = None

        self.account_session_id = account_session_id

        return True

    def _order_log_handler(self, order: dict, event: dict) -> bool:
        if not self.log_order:
            return False

        # update broker order status via rest API
        main_order_data = {}
        main_order_id = None
        order_data = {}
        order_id = None

        transaction_date = parse_transaction_date(event['transaction_date'])
        if transaction_date is None:
            logging.error(f'Invalid transaction date returned: {event["transaction_date"]}')
            return False
        transaction_date_est = get_est_time(transaction_date)

        if event['status'] == 'filled':
            if order['order_type'] in ['LMT', 'STP']:  # limit/stop order filled, main order closed
                main_order_id = order['parent_order_id']
                main_order_data = {
                    'status': 'CL',
                    'closed_time': transaction_date_est.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'closed_by': 'TF' if order['order_type'] == 'LMT' else 'SL'
                }
                order_id = order['id']
                order_data = {
                    'status': 'FL',
                    'filled_time': transaction_date_est.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'filled_price': event['avg_fill_price']
                }
            elif order['order_type'] == 'MKT':
                if order['parent_order_id']:    # main order rollback, closed
                    main_order_id = order['parent_order_id']
                    main_order_data = {
                        'status': 'CL',
                        'closed_time': transaction_date_est.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        'closed_by': 'FC'
                    }
                    order_id = order['id']
                    order_data = {
                        'status': 'FL',
                        'filled_time': transaction_date_est.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        'filled_price': event['avg_fill_price']
                    }
                else:   # main order filled
                    main_order_id = order['id']
                    main_order_data = {
                        'status': 'FL',
                        'filled_time': transaction_date_est.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        'filled_price': event['avg_fill_price']
                    }

        elif event['status'] == 'canceled':
            if order['order_type'] == 'MKT' and not order['parent_order_id']:  # main order has been cancelled
                main_order_id = order['id']
                main_order_data = {
                    'status': 'CL',
                    'closed_time': transaction_date_est.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'closed_by': 'FC'
                }
            else:
                order_id = order['id']
                order_data = {
                    'status': 'CN',
                    'closed_time': transaction_date_est.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                }

        # update main order status
        if main_order_data and main_order_id:
            self.sim_req.patch_response(f'{self.rest_api_orders_url}/{main_order_id}', main_order_data)

        # update child order status
        if order_data and order_id:
            self.sim_req.patch_response(f'{self.rest_api_orders_url}/{order_id}', order_data)

        return False

    def _open_tp_sl_orders(self, order: dict, event: dict) -> bool:
        # get the latest active positions
        self.get_active_positions()

        # try to get child orders under this order
        active_main_orders = self._get_main_orders(id=order['id'])
        if len(active_main_orders) == 0:
            return False

        # check if this main_order already has limit/stop order or not
        m_order = active_main_orders[0]
        sub_orders = m_order['child_orders']
        for s_order in sub_orders:
            if s_order['order_type'] in ['LMT', 'STP']:
                # this main order already has limit/stop
                return False

        # prepare parameters
        main_order_id = str(m_order['order_id'])
        symbol = m_order['symbol']
        qty = m_order['quantity']
        duration = 'day'
        main_order_price = float(m_order['filled_price'])
        snp_pct = m_order['snapshot']
        take_profit_price = m_order['take_profit_price']
        stop_loss_price = m_order['stop_loss_price']

        # determine order side
        if m_order['order_side'] == 'BUY':
            side_count = 1
            ls_order_side = 'sell'
        elif m_order['order_side'] == 'SELL':
            side_count = -1
            ls_order_side = 'buy'
        elif m_order['order_side'] == 'SELL SHORT':
            side_count = -1
            ls_order_side = 'buy_to_cover'
        else:
            return False

        # current owned shares
        if symbol in self.positions:
            own_shares = self.positions[symbol]['quantity']
        else:
            own_shares = 0

        open_stop_order = True
        open_limit_order = True
        if ls_order_side == 'sell' and own_shares < qty * 2:
            # if don't have enough shares to open both of stop and limit order, only open stop order.
            open_limit_order = False
        if ls_order_side == 'buy_to_cover':
            open_limit_order = False

        ls_order_params = {
            'class': 'equity',
            'symbol': symbol,
            'side': ls_order_side,
            'quantity': str(qty),
            'duration': duration
        }

        # open stop order
        if open_stop_order:
            stop = str(round(main_order_price * ((stop_loss_price * -1 * side_count) + 1), 2))
            stop_order_params = ls_order_params.copy()
            stop_order_params['type'] = 'stop'
            stop_order_params['stop'] = stop
            stop_order_params['tag'] = f'{symbol}-{snp_pct}-{main_order_id}-equity-stop'
            self._open_tradier_order(stop_order_params)

        # open limit order
        if open_limit_order:
            price = str(round(main_order_price * ((take_profit_price * 1 * side_count) + 1), 2))
            limit_order_params = ls_order_params.copy()
            limit_order_params['type'] = 'limit'
            limit_order_params['price'] = price
            limit_order_params['tag'] = f'{symbol}-{snp_pct}-{main_order_id}-equity-limit'
            self._open_tradier_order(limit_order_params)

        return True

    def _cancel_unnecessary_ls_order(self, filled_order: dict, event: dict) -> bool:
        """
        Since we have two additional orders (limit, stop) for each main market order.
        When one of them has been filled, we need to cancel another to prevent double-buy, or double-sell
        """
        # get the latest active positions
        self.get_active_positions()

        # find main order
        main_order_id = filled_order['parent_order_id']
        main_orders = self._get_main_orders(id=main_order_id)
        if len(main_orders) == 0:
            return False
        main_order = main_orders[0]

        # cancel unnecessary limit/stop order in pending status
        for sub_order in main_order['child_orders']:
            if sub_order['order_type'] in ['LMT', 'STP'] and sub_order['status'] == 'OP':
                self._cancel_tradier_order(sub_order['order_id'])

        return True

    def _order_event_triggered(self, event) -> bool:
        """
        Order status has been changed. We get this event via websocket.
        """
        # First, get order data from event info
        resp = self.sim_req.get_response(
            f'{self.rest_api_orders_url}/?broker={self.broker_nickname}&order_id={event["id"]}&'
            f'client_id={self.account_id}',
            attach_slash=False
        )
        if resp is None:
            return False

        if resp.json():
            order = resp.json()[0]
        else:
            # if this is new order, will get the order info from tradier api and save it to db
            new_order = self.get_orders(event['id'])
            order = None
            # sometimes, "pending" and "open" events are triggered almost simultaneously. I will save order only
            # when "open" event has been triggered.
            if event['status'] == 'open':
                order = self._save_new_order(new_order)
            if not order:
                return False

        logging.info(f'Order event: {event} {order}')

        # if this order is not equity order, it might be OTO, OCO, OTOCO order, skip to process for now
        if event['tag'].split('-')[3] != 'equity':
            return False

        # Second, update broker order through rest API
        self._order_log_handler(order, event)

        # Third, treat order
        try:
            if order['order_type'] in ['LMT', 'STP'] and event['status'] == 'filled':
                # if limit/stop order has been filled, need to cancel the related stop/limit order to prevent double-
                # sell or double-buy
                return self._cancel_unnecessary_ls_order(order, event)

            elif order['order_type'] in ['MKT'] and event['status'] == 'filled' and \
                    order['order_side'] != 'BUY TO COVER':
                if order['parent_order_id']:   # if not main order, skip.
                    return False
                # main order has been filled, need to open tp/sl orders now.
                return self._open_tp_sl_orders(order, event)

        except Exception as e:
            logging.error(f'Parse {event["id"]} order event failed: {e}')
            return False

        return False

    async def _ws_connect_and_consume(self):
        websocket_endpoint_url = '/v1/accounts/events'

        async with websockets.connect(self.websocket_base_url + websocket_endpoint_url) as websocket:
            logging.info(f'websocket connection published')
            payload = '{"events": ["order"], "sessionid": "' + self.account_session_id + '", "excludeAccounts": []}'

            await websocket.send(payload)

            while True:
                response = await websocket.recv()
                event = json.loads(response)
                if event['event'] == 'order':
                    self._order_event_triggered(event)
                elif event['event'] == 'heartbeat':
                    # {'event': 'heartbeat', 'status': 'active', 'timestamp': '2023-11-30T18:44:08.005364094Z'}
                    continue

    def _run_streaming_order_event(self):
        if self.account_session_id is None:
            res = self._create_account_session()
            if not res:
                return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ws_connect_and_consume())

    def _open_equity_order(self, forecast_id: str, snp_pct: int, order_leg: OrderLeg) -> bool:
        try:
            symbol = order_leg.symbol
            quantity = order_leg.quantity
            order_type = order_leg.order_type
            order_side = order_leg.order_side
            current_price = order_leg.cp
            forecasted_price = order_leg.fp
            take_profit_price = order_leg.tp        # this value is the percentage
            stop_loss_price = order_leg.sl          # this value is the percentage
            duration = order_leg.duration

            # determine order type
            type_decision = self._get_order_type_string(order_type)

            # determine the order side again according to the positions and owned shares qty
            side_decision = self._get_order_side_string(order_side)
            if side_decision == 'sell':
                if symbol not in self.positions or self.positions[symbol]['quantity'] < quantity:
                    side_decision = 'sell_short'

            # open equity order
            params = {
                'class': self._get_order_class_string(TradingOrderType.Equity),
                'symbol': symbol,
                'side': side_decision,
                'quantity': str(quantity),
                'type': type_decision,
                'duration': duration,
                'price': take_profit_price,
                'stop': stop_loss_price,
                'tag': f'{symbol}-{snp_pct}-0-equity-{type_decision.replace("_", "")}'
            }
            order_id = self._open_tradier_order(params)
            if order_id is None:
                return False
            logging.info(f'Place an market order for {symbol}: success {order_id}')

            # store tp/sl values to open tp/sl orders later
            self.orders_tp_sl_map[str(order_id)] = {
                'forecast_id': forecast_id,
                'current_price': current_price,
                'forecast_price': forecasted_price,
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price
            }

        except Exception as e:
            logging.error(f'{order_leg.symbol} cannot be placed an order, {e}')
            return False

        return True

    def _open_combined_orders(self, forecast_id: str, snp_pct: int, trading_order_type: TradingOrderType,
                              order_legs: list[OrderLeg]) -> bool:
        order_class = self._get_order_class_string(trading_order_type)
        params = {
            'class': order_class,
            'tag': f'0-{snp_pct}-0-{order_class}-0'
        }
        for i in range(len(order_legs)):
            order_leg = order_legs[i]
            params[f'symbol[{i}]'] = order_leg.symbol
            params[f'quantity[{i}]'] = str(order_leg.quantity)
            params[f'type[{i}]'] = self._get_order_type_string(order_leg.order_type)
            params[f'option_symbol[{i}]'] = order_leg.option_symbol
            params[f'side[{i}]'] = self._get_order_side_string(order_leg.order_side)
            params[f'duration[{i}]'] = order_leg.duration
            params[f'price[{i}]'] = str(order_leg.tp)
            params[f'stop[{i}]'] = str(order_leg.sl)

        order_id = self._open_tradier_order(params)
        if order_id is None:
            return False

        return True

    def open_order(self, forecast_id: str, snp_pct: int, trading_order_type: TradingOrderType,
                   order_legs: list[OrderLeg]) -> bool:
        if trading_order_type == TradingOrderType.Equity:
            return self._open_equity_order(forecast_id, snp_pct, order_legs[0])
        else:
            return self._open_combined_orders(forecast_id, snp_pct, trading_order_type, order_legs)

    def close_order(self, dt, snp_pct: int, symbol: str) -> bool:
        main_orders = self._get_main_orders(date=dt.strftime('%Y-%m-%d'), snapshot=snp_pct, symbol=symbol)

        # filter orders that we need to cancel or rollback orders
        need_cancel_orders = []
        need_rollback_orders = []
        for m_order in main_orders:
            if m_order['status'] == 'OP':
                need_cancel_orders.append(m_order)

            rollback_order_flag = False
            for sub_order in m_order['child_orders']:
                if sub_order['status'] == 'OP':
                    need_cancel_orders.append(sub_order)
                if sub_order['order_type'] == 'MKT':
                    rollback_order_flag = True

            # To protect rollback twice for one parent order, need to check rollback_order_flag
            if m_order['status'] == 'FL' and rollback_order_flag is False:
                need_rollback_orders.append(m_order)

        # cancel orders
        for order in need_cancel_orders:
            self._cancel_tradier_order(order['order_id'])

        # rollback orders
        for order in need_rollback_orders:
            if order['order_side'] == 'BUY':
                order_side = 'sell'
            elif order['order_side'] == 'SELL':
                order_side = 'buy'
            elif order['order_side'] == 'SELL SHORT':
                order_side = 'buy_to_cover'
            else:
                continue

            params = {
                'class': 'equity',
                'symbol': order['symbol'],
                'side': order_side,
                'quantity': str(order['quantity']),
                'type': 'market',
                'duration': 'day',
                'tag': f'{order["symbol"]}-{order["snapshot"]}-{order["order_id"]}-equity-market'
            }
            self._open_tradier_order(params)

        return True

    def get_active_positions(self) -> dict:
        active_pos = dict()

        '''
        Open all the active positions per symbol
        '''
        endpoint_url = f'/v1/accounts/{self.account_id}/positions'
        params = {}

        response = requests.get(self.base_url + endpoint_url,
                                params=params,
                                headers=self.bearer_headers
                                )
        open_positions = response.json()

        if open_positions['positions'] != 'null':
            position = open_positions['positions']['position']
            current_positions = []
            if type(position) is dict:
                current_positions = [position]

            elif type(position) is list:
                current_positions = position

            for pos in current_positions:
                active_pos[pos['symbol']] = pos

        self.positions = active_pos
        return active_pos

    def add_funds(self, amount: float) -> bool:
        endpoint_url = f'/v2/virtual/accounts/{self.account_id}/funding'
        response = requests.put(
            self.base_url + endpoint_url,
            json={'amount': amount},
            headers=self.basic_headers
        )
        if response.status_code >= 300:
            return False

        return True

    def get_total_balance(self) -> float:
        endpoint_url = f'/v1/accounts/{self.account_id}/balances'
        response = requests.get(
            self.base_url + endpoint_url,
            params={},
            headers=self.bearer_headers
        )
        if response.status_code != 200:
            return 0

        json_response = response.json()
        try:
            total_cash = json_response['balances']['total_cash']
        except Exception as e:
            logging.error(f'Failed to get balance, {e}')
            total_cash = 0

        return total_cash

    def get_orders(self, order_id: int = 0):
        endpoint_url = f'/v1/accounts/{self.account_id}/orders'
        all_orders = []
        page_number = 1

        # fetch all orders
        while True:
            response = requests.get(
                self.base_url + endpoint_url,
                params={'page': str(page_number), 'includeTags': 'true'},
                headers=self.bearer_headers
            )

            if response.status_code != 200:
                break

            try:
                json_response = response.json()
                if json_response['orders'] == 'null':
                    break

                orders = json_response['orders']['order']

                if type(orders) is dict:
                    all_orders.append(orders)

                elif type(orders) is list:
                    all_orders.extend(orders)

                page_number += 1

            except Exception as e:
                logging.error(f'Cannot get orders, {e}')
                break

        # find the specific order
        if order_id > 0:
            for order in all_orders:
                if order['id'] == order_id:
                    return order
            return None

        return all_orders

    def create_virtual_accounts(self):
        endpoint_url = '/v2/virtual/accounts'
        self.basic_headers = {
            'Authorization': 'Basic Ulpkd0N5NmdvUW9KSllNZjdkR3lNbkh5V3l4RzJBN0I6bnpVeEl2QWVMeDRVSUNpRQ==',
            'Accept': 'application/json'
        }

        params = {'firstName': 'Nicky', 'lastName': 'Sayouth', 'email': 'nicky@tsgs.com', 'agreementSigned': aslocaltimestr(datetime.now()), 'repCode': 'TA'}

        response = requests.post(self.base_url + endpoint_url,
                                 json=params,
                                 headers=self.basic_headers
                                 )

        if response.status_code != 200:
            return

        json_response = response.json()
        self.account_id = json_response['accountNumber']
        self.bearer_headers = {'Authorization': f'Bearer {json_response["token"]}', 'Accept': 'application/json'}
