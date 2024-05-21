# multi-video play
import numpy as np
import pandas as pd
import math

VIDEO_BIT_RATE = [750,1200,1850]  # Kbps
VIDEO_CHUNCK_LEN = 1  # s
VIDEO_FILE = 'dataset/mod_video_info.csv'

class Player:
    # initialize each new video and player buffer
    def __init__(self, video_id):  
        # initialize each new video
        video_data = pd.read_csv(VIDEO_FILE)
        mod_video_id = video_data[video_data['Video Id'] == video_id].iloc[0]['id']
        self.current_video_data = video_data[video_data['id'] == mod_video_id]
        # print("debug:", self.current_video_data)

        # process kuaishou videos(only get 720p)
        if self.current_video_data.shape[0] == 1:
            # 复制一行数据成3行
            self.current_video_data = pd.concat([self.current_video_data] * 3, ignore_index=True)

            # 将 "Bitrate" 列的值分别改为 0, 1, 2
            self.current_video_data['Bitrate'] = [0, 1, 2] * 1
            # print(current_video_data)

            # 定义一个函数，用于处理每个元素
            def process_kuaishou_video(chunk_size, ratio):
                if pd.notna(chunk_size):
                    return int(chunk_size * ratio)
                else:
                    return chunk_size

            # 对第六列开始的每个元素应用处理函数
            columns_to_process = self.current_video_data.columns[5:]
            self.current_video_data.loc[self.current_video_data['Bitrate'] == 0, columns_to_process] \
                = self.current_video_data.loc[self.current_video_data['Bitrate'] == 0, columns_to_process \
                ].map(lambda x: process_kuaishou_video(x, 0.8))

            self.current_video_data.loc[self.current_video_data['Bitrate'] == 2, columns_to_process] \
                = self.current_video_data.loc[self.current_video_data['Bitrate'] == 2, columns_to_process \
                ].map(lambda x: process_kuaishou_video(x, 1.2))

        # print(self.current_video_data)
        # num and len of the video, all chunks are counted instead of -1 chunk
        if self.current_video_data.iloc[0]["Duration"] != self.current_video_data.iloc[1]["Duration"] \
            or self.current_video_data.iloc[0]["Duration"] != self.current_video_data.iloc[2]["Duration"]:
            print("wrong video data, video id: ", video_id)
            adjust_duration = min(self.current_video_data.iloc[0]["Duration"], \
                                    self.current_video_data.iloc[1]["Duration"], \
                                    self.current_video_data.iloc[2]["Duration"])
            self.duration = adjust_duration
        else:
            self.duration = self.current_video_data.iloc[0]["Duration"]
        self.chunk_num = self.duration / VIDEO_CHUNCK_LEN

        # content of the video
        self.content = self.current_video_data.iloc[2]["Content"]

        # download chunk counter(id of chunk to be downloaded)
        self.download_chunk_counter = 0
        self.download_chunk_remain = self.chunk_num - self.download_chunk_counter
        
        self.download_bitrate_record = []
        
        # play chunk counter
        self.play_chunk_counter = 0
        
        # play timeline of this video
        self.play_timeline = 0.0
        
        # initialize the buffer
        self.buffer_size = 0  # s
        
        # initialize preload size
        self.preload_size = 0 # B

        # initialize waste size & QoE & reward of this video
        self.waste_size = 0 # B
        self.QoE = 0
        self.reward = float('-inf')

        self.video_id = video_id

        # initialize corresponding user watch time



    def get_chunk_size(self, chunk_id, bitrate):
        try:
            # +5 means the first 5 columns in video_data_test.csv
            chunk_size = self.current_video_data[self.current_video_data['Bitrate'] == bitrate].iloc[0, chunk_id+5]
        except IndexError:
            # raise Exception("You're downloading chunk ["+str(chunk_id)+"] is out of range. "\
            #                 + "\n   % Hint: The valid chunk id is from 0 to " + str(self.chunk_num-1) + " %")
            return 0
        return chunk_size

    def get_download_chunk_bitrate(self, chunk_id):
        if chunk_id >= len(self.download_bitrate_record):
            return -1  # means this chunk hasn't been downloaded yet
        return self.download_bitrate_record[chunk_id]

    def record_download_bitrate(self, bitrate):
        self.download_bitrate_record.append(bitrate)
        self.preload_size += self.get_chunk_size(self.download_chunk_counter, bitrate)

    def record_redundant_waste(self, wastage):
        self.waste_size += wastage

    def bandwidth_waste(self, watch_duration):
        download_len = len(self.download_bitrate_record) # start from 1
        waste_start_chunk = math.ceil(watch_duration)
        sum_waste = 0
        for i in range(waste_start_chunk, download_len):
            download_bitrate = self.download_bitrate_record[i]
            download_size = self.get_chunk_size(i, download_bitrate)
            sum_waste += download_size
        self.waste_size += sum_waste
        return sum_waste

    # download the video, buffer increase.
    def video_download(self, download_len):  # s
        self.buffer_size += download_len
        self.download_chunk_counter += 1
        self.download_chunk_remain = self.chunk_num - self.download_chunk_counter
        video_download_complete = False
        if self.download_chunk_counter >= self.chunk_num:
            video_download_complete = True
        return video_download_complete

    # play the video, buffer decrease. Return the remaining buffer, negative number means rebuf
    def video_play(self, play_time):  # s
        buffer = round(self.buffer_size - play_time, 3)
        # print("test1", self.buffer_size, play_time, self.play_timeline)
        self.play_timeline += np.minimum(self.buffer_size, play_time)   # rebuffering time is not included in timeline
        self.play_timeline = round(self.play_timeline, 3)
        # print("test2")
        self.buffer_size = round(np.maximum(self.buffer_size - play_time, 0.0), 3)
        # print(self.buffer_size, play_time, "play_timeline", self.play_timeline)
        return self.play_timeline, buffer

    def record_QoE_reward(self, QoE, reward):
        self.QoE = QoE
        self.reward = reward
