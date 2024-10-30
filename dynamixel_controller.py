import threading
from dynamixel_sdk import *  # Dynamixel SDKのインポート
import serial

# 定数
ADDR_XL_TORQUE_ENABLE = 64
ADDR_XL_PRESENT_LOAD = 126  # Loadのアドレス
ADDR_XL_PRESENT_VELOCITY = 128  # Velocityのアドレス
ADDR_XL_PRESENT_POSITION = 132  # Positionのアドレス
ADDR_XL_GOAL_POSITION = 116  # 使用しているモデルに応じて変更
ADDR_XL_GOAL_VELOCITY = 104  # 速度設定用アドレス
ADDR_XL_GOAL_PWM = 100
ADDR_XL_GOAL_PWM_LIMIT = 36
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0 
COMM_SUCCESS = 0

port_lock = threading.Lock()

def convert_to_signed(value, bits):
    """
    符号なし整数を符号付き整数に変換します。

    :param value: 符号なし整数値
    :param bits: ビット数（例：16, 32）
    :return: 符号付き整数値
    """
    if value >= 2**(bits - 1):
        value -= 2**bits
    return value

class DynamixelController:
    def __init__(self, device_name, baudrate=1000000):
        self.baudrate = baudrate
        self.device_name = device_name
        self.portHandler = PortHandler(device_name)
        self.packetHandler = PacketHandler(2.0)  # Ensure the correct protocol version is passed

    def setup_port(self):
        with port_lock:
            try:
                if not self.portHandler.openPort():
                    print(f"--------------------")
                    print("接続失敗")
                    print(f"--------------------")
                    print(f"Device名: {self.device_name}")
                    print(f"Baudrate: {self.baudrate}")
                    print(f"--------------------")
                    return False
                if not self.portHandler.setBaudRate(self.baudrate):
                    print(f"--------------------")
                    print("ポート設定エラー")
                    print(f"--------------------")
                    print(f"Device名: {self.device_name}")
                    print(f"Baudrate: {self.baudrate}")
                    print(f"--------------------")
                    return False
            except (FileNotFoundError, serial.SerialException) as e:
                print(f"--------------------")
                print("接続失敗")
                print(f"--------------------")
                print(f"Device名: {self.device_name}")
                print(f"Baudrate: {self.baudrate}")
                print(f"エラー内容: {e}")
                print(f"--------------------")
                return False

            print(f"--------------------")
            print("接続成功")
            print(f"--------------------")
            print(f"Device名: {self.device_name}")
            print(f"Baudrate: {self.baudrate}")
            print(f"--------------------")
            return True

    def enable_torque(self, ids):
        with port_lock:
            for dxl_id in ids:
                dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, ADDR_XL_TORQUE_ENABLE, TORQUE_ENABLE)
                
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"ID {dxl_id} のトルクを有効にすることに失敗しました: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
                elif dxl_error != 0:
                    print(f"ID {dxl_id} のトルク有効化中にエラーが発生しました: {self.packetHandler.getRxPacketError(dxl_error)}")
                else:
                    print(f"ID: {dxl_id} のトルクオプションを有効にしました。")

    def disable_torque(self, ids):
        """
        指定されたIDのDynamixelサーボモーターのトルクを無効にします。

        :param ids: サーボモーターのIDリスト
        """
        with port_lock:
            for dxl_id in ids:
                dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, ADDR_XL_TORQUE_ENABLE, TORQUE_DISABLE)
                
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"ID {dxl_id} のトルクを無効にすることに失敗しました: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
                elif dxl_error != 0:
                    print(f"ID {dxl_id} のトルク無効化中にエラーが発生しました: {self.packetHandler.getRxPacketError(dxl_error)}")
                else:
                    print(f"ID: {dxl_id} のトルクオプションを無効にしました。")


    def set_operation_mode(self, ids, mode):
        """
        指定されたIDのDynamixelサーボモーターを目的の動作モードに設定する。
        
        :param ids: サーボモーターのIDリスト
        :param mode: 設定する動作モード（例: 1: Velocity Mode, 16: PWM Mode）
        """
        ADDR_OPERATING_MODE = 11  # 動作モードのアドレス（モデルに応じて変更が必要）

        with port_lock:
            for dxl_id in ids:
                # 動作モードを設定
                result, error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, ADDR_OPERATING_MODE, mode
                )
                if result != COMM_SUCCESS:
                    print(f"ID {dxl_id} のモード設定に失敗しました: {self.packetHandler.getTxRxResult(result)}")
                elif error != 0:
                    print(f"ID {dxl_id} のモード設定中にエラーが発生しました: {self.packetHandler.getRxPacketError(error)}")
                else:
                    # 設定が成功した場合、モードの確認
                    current_mode, _, _ = self.packetHandler.read1ByteTxRx(
                        self.portHandler, dxl_id, ADDR_OPERATING_MODE
                    )
                    if current_mode == mode:
                        print(f"ID {dxl_id} のモードを {mode} に設定しました。")
                    else:
                        print(f"ID {dxl_id} のモード設定が反映されていません。")

    def sync_write_goal_position(self, ids, goal_positions, address, data_length=4):
        with port_lock:
            group_sync_write = GroupSyncWrite(self.portHandler, self.packetHandler, address, data_length)
            for i, dxl_id in enumerate(ids):
                goal_position = goal_positions[i]
                data = [
                    DXL_LOBYTE(DXL_LOWORD(goal_position)),
                    DXL_HIBYTE(DXL_LOWORD(goal_position)),
                    DXL_LOBYTE(DXL_HIWORD(goal_position)),
                    DXL_HIBYTE(DXL_HIWORD(goal_position))
                ]
                add_param_result = group_sync_write.addParam(dxl_id, data)
                if not add_param_result:
                    print(f"ID {dxl_id} のパラメータ追加に失敗しました。")
                    return False
            
            # サーボにパケットを送信
            dxl_comm_result = group_sync_write.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print(f"ゴールポジションの同期書き込みに失敗しました: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
                return False
            #else:
            #    print("全てのサーボに対してゴールポジションの書き込みが成功しました。")
            group_sync_write.clearParam()
            return True

    def sync_read_data(self, ids):
        with port_lock:
            group_sync_read = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_XL_PRESENT_LOAD, 10)
            
            for dxl_id in ids:
                if not group_sync_read.addParam(dxl_id):
                    print(f"ID {dxl_id} のパラメータ読み込み追加に失敗しました。")
                    return False

            # Read the data
            dxl_comm_result = group_sync_read.txRxPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print(f"データの同期読み込みに失敗しました: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
                return False

            results = {}
            for dxl_id in ids:
                load_data = group_sync_read.getData(dxl_id, ADDR_XL_PRESENT_LOAD, 2)
                load = convert_to_signed(load_data, 16)  # 16ビットの符号付き整数に変換

                velocity_data = group_sync_read.getData(dxl_id, ADDR_XL_PRESENT_VELOCITY, 4)
                velocity = convert_to_signed(velocity_data, 32)  # 32ビットの符号付き整数に変換

                position = group_sync_read.getData(dxl_id, ADDR_XL_PRESENT_POSITION, 4)
                results[dxl_id] = {'load': load, 'velocity': velocity, 'position': position}
                #print(f"ID: {dxl_id}, Load: {load}, Velocity: {velocity}, Position: {position}")

            group_sync_read.clearParam()
            return results

    def set_pwm(self, ids, pwms):
        """
        DynamixelをPWM Modeで制御します。
        
        :param ids: サーボモーターのIDリスト
        :param pwms: 各サーボモーターに対応する目標PWMのリスト（正で正回転、負で逆回転）
        """
        if len(ids) != len(pwms):
            print("エラー: 'ids' と 'pwms' のリストの長さが一致しません。")
            return

        with port_lock:
            for i, dxl_id in enumerate(ids):
                try:
                    goal_pwms = pwms[i]
                except IndexError:
                    print(f"ID {dxl_id} のPWM設定中にエラーが発生しました: インデックスエラー")
                    continue
                
                # PWMデータの整形
                pwm_data = [
                    DXL_LOBYTE(DXL_LOWORD(goal_pwms)),
                    DXL_HIBYTE(DXL_LOWORD(goal_pwms))
                ]
                
                # PWMを設定
                dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                    self.portHandler, dxl_id, ADDR_XL_GOAL_PWM, goal_pwms)
                
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"ID {dxl_id} のPWM設定に失敗しました: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
                elif dxl_error != 0:
                    print(f"ID {dxl_id} のPWM設定中にエラーが発生しました: {self.packetHandler.getRxPacketError(dxl_error)}")
                else:
                    print(f"ID: {dxl_id} のPWMを {goal_pwms} に設定しました。")

    def set_velocity(self, ids, velocities):
        """
        DynamixelをVelocity Modeで制御します。
        
        :param ids: サーボモーターのIDリスト
        :param velocities: 各サーボモーターに対応する目標速度のリスト（正で正回転、負で逆回転）
        """
        if len(ids) != len(velocities):
            print("エラー: 'ids' と 'velocities' のリストの長さが一致しません。")
            return

        with port_lock:
            for i, dxl_id in enumerate(ids):
                try:
                    goal_velocity = velocities[i]
                except IndexError:
                    print(f"ID {dxl_id} の速度設定中にエラーが発生しました: インデックスエラー")
                    continue
                
                # 速度データの整形
                velocity_data = [
                    DXL_LOBYTE(DXL_LOWORD(goal_velocity)),
                    DXL_HIBYTE(DXL_LOWORD(goal_velocity)),
                    DXL_LOBYTE(DXL_HIWORD(goal_velocity)),
                    DXL_HIBYTE(DXL_HIWORD(goal_velocity))
                ]
                
                # 速度を設定
                dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                    self.portHandler, dxl_id, ADDR_XL_GOAL_VELOCITY, goal_velocity)
                
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"ID {dxl_id} の速度設定に失敗しました: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
                elif dxl_error != 0:
                    print(f"ID {dxl_id} の速度設定中にエラーが発生しました: {self.packetHandler.getRxPacketError(dxl_error)}")
                else:
                    print(f"ID: {dxl_id} の速度を {goal_velocity} に設定しました。")
