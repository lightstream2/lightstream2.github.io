import numpy as np
import pandas as pd
import math
from video_player import Player
from network_module import Network

NEW = 0
DEL = 1

VIDEO_BIT_RATE = [750, 1200, 1850]  # Kbps
B_IN_MB = 1000000.0
VIDEO_CHUNCK_LEN = 1 # s
PAST_VIDEO_NUM = 10
PLAYER_NUM = 5
FUTURE_CHUNK_WEIGHT = [0.7, 0.2, 0.1]
THRESHOLD_DURATION = 60  # s
QOE_BETA = 0.02
THRESHOLD_THROUGHPUT = 375000  # B/s(3Mbps)
THRESHOLD_WATCH_PERCENT = 0.5
THRESHOLD_WATCH_DURATION = 10
BITRATE_BETA = 0.05
THRESHOLD_QOE = 0.5



class Environment:
    def __init__(self, session_id, net_trace_id, net_quality = 'mixed'):
        global SESSION_FILE, NET_TRACE_FILE
        NET_TRACE_FILE = 'dataset/net_trace/'+str(net_quality)+'/'+str(net_trace_id)+'.txt'
        SESSION_FILE = 'dataset/session/'+str(session_id)+'.csv'

        with open(NET_TRACE_FILE, 'r') as file:
            lines = file.readlines()
        net_time = [line.split()[0] for line in lines]
        net_bandwidth = [line.split()[1] for line in lines]

        session_data = pd.read_csv(SESSION_FILE)
        # total video num in this session(start from 1)
        self.video_num = session_data.shape[0]
        self.video_id_arr = session_data['videoId'].values
        # print(self.video_id_arr)
        session_data['timeSpent'] = round(session_data['timeSpent'])
        self.watch_duration_arr = session_data['timeSpent'].values
        # num of videos user watched(start from 0)看了一个视频是0
        self.watch_num = np.flatnonzero(np.array(self.watch_duration_arr))[::-1][0]
        # next video count to insert into the playback queue
        self.list_cnt = 0
        # current playing video count
        self.play_cnt = 0
        self.network = Network(net_time, net_bandwidth)
        # self.timeline = 0.0
        # num of videos that users are seriously watching
        self.watched_video_num = 0
        self.total_waste = 0  # B
        self.reduce_wasted = 0  # B
        self.total_QoE = 0
        self.total_reward = 0

        self.players = [] # play list(1 playing & 4 to play)
        self.past_players = [] # 10 videos user just watched(Reverse order)
        for p in range(PLAYER_NUM):
            video_id = self.video_id_arr[p]
            self.players.append(Player(video_id))
            self.list_cnt += 1

    def player_op(self, operation):
        if operation == NEW:
            # print('--------------ADD--------------')
            if self.list_cnt >= self.video_num:  # If exceed video cnt, no add
                return
            video_id = self.video_id_arr[self.list_cnt]
            self.players.append(Player(video_id))
            # print("debug2", self.players[4].video_id, self.players[4].download_chunk_counter)
            # print("adding: video", video_id)
            self.list_cnt += 1
        else:
            # print('--------------DEL--------------')
            self.past_players = [self.players[0]] + self.past_players
            if len(self.past_players) > 10:
                self.past_players.remove(self.past_players[10])
            self.players.remove(self.players[0])

    def judge_network(self):
        past_throughput = self.network.past_throughput
        count_bad_throughput = sum(throughput < THRESHOLD_THROUGHPUT for throughput in past_throughput)
        mean_throughput = sum(past_throughput) / len(past_throughput)
        if count_bad_throughput >= 5 or mean_throughput < THRESHOLD_THROUGHPUT:
            return 0  # bad network
        else:
            return 1  # good network
        
    def judge_bitrate(self):
        download_bitrate_record = self.players[0].download_bitrate_record
        bitrate_0_ratio = download_bitrate_record.count(0) / len(download_bitrate_record)
        bitrate_1_ratio = download_bitrate_record.count(1) / len(download_bitrate_record)
        bitrate_2_ratio = download_bitrate_record.count(2) / len(download_bitrate_record)
        net_level = self.judge_network()
        if net_level == 0:
            bitrate_0_coefficient = 1
            bitrate_1_coefficient = 1 + BITRATE_BETA
            bitrate_2_coefficient = 1 + BITRATE_BETA * 2
        else:
            bitrate_0_coefficient = 1 - BITRATE_BETA
            bitrate_1_coefficient = 1
            bitrate_2_coefficient = 1 + BITRATE_BETA
        bitrate_coefficient = bitrate_0_ratio * bitrate_0_coefficient \
                            + bitrate_1_ratio * bitrate_1_coefficient \
                            + bitrate_2_ratio * bitrate_2_coefficient
        decrease_punishment = 0
        for i in range(1, len(download_bitrate_record)):
            if download_bitrate_record[i] < download_bitrate_record[i - 1]:
                decrease_punishment += download_bitrate_record[i - 1] - download_bitrate_record[i]
        bitrate_coefficient -= decrease_punishment * BITRATE_BETA
        return bitrate_coefficient

    def get_QoE(self):
        watch_duration = self.watch_duration_arr[self.play_cnt]
        video_duration = self.players[0].duration
        mod_watch_percent = min(1, watch_duration / min(video_duration, THRESHOLD_DURATION))
        bitrate_coefficient = self.judge_bitrate()
        persistent_watch_factor = pow(1 + QOE_BETA, self.watched_video_num)
        # print("mod_watch_percent", mod_watch_percent)
        # print("persistent_watch_factor", persistent_watch_factor)
        # print("bitrate_coefficient", bitrate_coefficient)
        QoE = mod_watch_percent * persistent_watch_factor * bitrate_coefficient
        return QoE
    
    def get_reward(self, QoE, wastage):
        if QoE >= THRESHOLD_QOE:
            alpha = 0
        else:
            alpha = 1
        reward = -(wastage / B_IN_MB + alpha) * max(THRESHOLD_QOE / QoE, 1)
        return reward

    def get_past_video_info(self):
        past_content = []
        past_mod_watch_percent = []
        for i in range(len(self.past_players)):
            content = self.past_players[i].content
            past_content.append(content)

            watch_duration = self.past_players[i].play_timeline
            video_duration = self.past_players[i].duration
            mod_watch_percent = min(1, watch_duration / min(video_duration, THRESHOLD_DURATION))
            past_mod_watch_percent.append(mod_watch_percent)

        # If number of videos watched is insufficient, padding -1
        if len(past_content) < PAST_VIDEO_NUM:
            padding_num = PAST_VIDEO_NUM - len(past_content)
            past_content = past_content + [-1] * padding_num
            past_mod_watch_percent = past_mod_watch_percent + [-1] * padding_num

        return past_content, past_mod_watch_percent
    
    def get_queue_video_info(self):
        queue_content = []
        queue_buffer_size = []
        queue_weighted_chunk_size = []
        for i in range(PLAYER_NUM):
            content = self.players[i % len(self.players)].content
            queue_content.append(content)

            buffer_size = self.players[i % len(self.players)].buffer_size
            queue_buffer_size.append(buffer_size)

            weighted_chunk_size = 0
            download_chunk_counter = self.players[i % len(self.players)].download_chunk_counter
            if download_chunk_counter == 0:
                bitrate = 0
            else:
                bitrate = self.players[i % len(self.players)].get_download_chunk_bitrate(download_chunk_counter-1)
            for j in range(len(FUTURE_CHUNK_WEIGHT)):
                # print("test11", i, download_chunk_counter+j, bitrate)
                chunk_size = self.players[i % len(self.players)].get_chunk_size(download_chunk_counter+j, bitrate)
                if np.isnan(chunk_size):
                    chunk_size = 0
                # print("test", chunk_size)
                weighted_chunk_size += chunk_size * FUTURE_CHUNK_WEIGHT[j]
            queue_weighted_chunk_size.append(round(weighted_chunk_size))

        return queue_content, queue_buffer_size, queue_weighted_chunk_size
    
    def play_videos(self, action_time, download_video_id):  # play for action_time from the start of current players queue
        # print("\n\nPlaying Video ", self.video_id_arr[self.play_cnt])
        # wastage in this action_time(not video wastage)
        wasted_bandwidth = 0
        buffer = 0
        terminate_download = False
        done = False
        ended_video_num = 0  # 如果有视频播放完，re_download_video_id发生偏移

        # Continues to play if all the following conditions are satisfied:
        # 1) there's still action_time len
        # 2) the last video hasn't caused rebuf
        # 3) the video queue is not empty (will break inside the loop if its already empty)
        while buffer >= 0 and action_time > 0:
            watch_duration = self.watch_duration_arr[self.play_cnt]
            video_duration = self.players[0].duration
            adjust_watch_duration = min(watch_duration, video_duration)
            # print("time_left:", action_time)
            # the timeline of the current video before this play step
            timeline_before_play = self.players[0].play_timeline
            # print("timeline_before_play: ", timeline_before_play)
            # the remain time length of the current video
            video_remain_time = adjust_watch_duration - timeline_before_play
            video_remain_time = round(video_remain_time, 3)
            # print("video_remain_time: ", video_remain_time)
            # the maximum play time of the current video
            max_play_time = min(action_time, video_remain_time)
            # print("max_play_time: ", max_play_time)
            # timeline_after_play is the actual time when the play action ended( <=max_play_tm + before_play )
            timeline_after_play, buffer = self.players[0].video_play(max_play_time)
            # print("timeline_after_play: ", timeline_after_play)
            # print("buffer:", buffer)
            # the actual time length of this play action
            actual_play_time = timeline_after_play - timeline_before_play
            actual_play_time = round(actual_play_time, 3)
            # print("actual_play_time: ", actual_play_time)
            # consume the action_time
            # print("test1", action_time, actual_play_time)
            action_time -= actual_play_time
            action_time = round(action_time, 3)
            # print("test2", action_time)

            # if the current playing video has ended
            if actual_play_time == video_remain_time:
                play_video_id = self.video_id_arr[self.play_cnt]
                watch_percent = adjust_watch_duration / self.players[0].duration
                if watch_percent >= THRESHOLD_WATCH_PERCENT \
                    or adjust_watch_duration >= THRESHOLD_WATCH_DURATION:
                    self.watched_video_num += 1
                # Output: the downloaded time length, the total time length, the watch duration
                print("\nUser stopped watching Video ", play_video_id, "( ", self.players[0].duration, " s ) :")
                print("User watched for ", adjust_watch_duration, " s, you downloaded ", len(self.players[0].download_bitrate_record), " s.")

                # Output the bitrates of this video:
                download_bitrate_record = self.players[0].download_bitrate_record
                # print("Your downloaded bitrates are: ", download_bitrate_record)

                # use watch duration as an arg for the calculation of wasted_bandwidth of this current video
                wasted_bandwidth += self.players[0].bandwidth_waste(adjust_watch_duration)

                # calculate QoE of current video
                video_QoE = self.get_QoE()
                self.total_QoE += video_QoE

                # get total waste of current video
                video_wastage = self.players[0].waste_size

                # calculate reward of current video and save to player
                video_reward = self.get_reward(video_QoE, video_wastage)
                self.total_reward += video_reward
                self.players[0].record_QoE_reward(video_QoE, video_reward)

                # Forward the queue head to the next video
                self.player_op(DEL)
                self.play_cnt += 1
                ended_video_num += 1
                self.player_op(NEW)
                # print("debug3", self.players[4].video_id, self.players[4].download_chunk_counter)

                # When the user swipes away the current video, and the downloader is downloading subsequent chunk of the video
                if download_video_id == play_video_id:
                    terminate_download = True
                    break

            # print("download_video_id: ", download_video_id)
            # print("play_cnt: ", self.play_cnt, "watch_num: ", self.watch_num)
            if self.play_cnt > self.watch_num:
                # if it has come to the end of the user watched list
                print("played out!")
                done = True
                break

        if self.play_cnt > self.watch_num:
            # if it has come to the end of the user watched list
            print("played out!")
            done = True

        if buffer < 0:  # action ends because a video stuck(needs rebuffer)
            buffer = (-1) * action_time  # rebuf time is the remain action time(cause the player will stuck for this time too)
        return buffer, wasted_bandwidth, terminate_download, actual_play_time, done, ended_video_num

    def take_action(self, re_download_video_id, bitrate, sleep_time): # Relative video id(0~4)
        buffer = 0
        rebuf = 0
        terminate_download = False
        terminate_download_time = 0.0
        temp_throughput_arr = []
        temp_duration_arr = []
        end_of_video = False
        delay = 0.0  # download time / sleep time
        download_video_id = self.video_id_arr[self.play_cnt + re_download_video_id]
        chunk_id = self.players[re_download_video_id].download_chunk_counter
        chunk_size = 0
        # wasted_bytes = 0
        done = False
        reward = 0
        

        if sleep_time > 0:
            # sleep_time = int(sleep_time)
            delay = sleep_time
            buffer, wasted, terminate_download, terminate_download_time, done, ended_video_num = self.play_videos(sleep_time, download_video_id)
            terminate_download = False
            # reward -= 0.05
            # # Return the end flag for the current playing video
            # if self.play_cnt == self.watch_num:  # if user leaves
            #     end_of_video = True
        else:
            # print("download_video_id", download_video_id, chunk_id, self.players[re_download_video_id].chunk_num)
            # This video has been downloaded completely
            if self.players[re_download_video_id].download_chunk_remain == 0:
                # Download the last chunk of this video and treat it as a waste
                print("Redundant Downloads. Video id: ", download_video_id)
                chunk_size = self.players[re_download_video_id].get_chunk_size(chunk_id-1, bitrate)
                self.players[re_download_video_id].record_redundant_waste(chunk_size)
                delay, temp_throughput_arr, temp_duration_arr = self.network.network_simu(chunk_size)  # s
                # print("delay", delay)
                buffer, wasted, terminate_download, terminate_download_time, done, ended_video_num = self.play_videos(delay, download_video_id)
                wasted += chunk_size
            else:
                # print("debug:download_video_id", download_video_id, "play_cnt", self.play_cnt)
                # print("debug:", download_video_id, chunk_id, self.players[re_download_video_id].download_chunk_remain)
                chunk_size = self.players[re_download_video_id].get_chunk_size(chunk_id, bitrate)
                # print("the actual download size is:", chunk_size)
                self.players[re_download_video_id].record_download_bitrate(bitrate)
                # print("debug", self.players[re_download_video_id].video_id, self.players[re_download_video_id].download_bitrate_record)
                delay, temp_throughput_arr, temp_duration_arr = self.network.network_simu(chunk_size)  # s
                # print("the actual download delay is:", delay)
                # print("\n\n")
                # play_timeline, buffer = self.players[self.play_video_id - self.play_cnt].video_play(delay)
                buffer, wasted, terminate_download, terminate_download_time, done, ended_video_num = self.play_videos(delay, download_video_id)
                # print("debug4", re_download_video_id, ended_video_num, self.players[re_download_video_id-ended_video_num].video_id)
                if re_download_video_id-ended_video_num == -1:
                    # print("!!!!!!!!!!")
                    video_download_complete = self.past_players[0].video_download(VIDEO_CHUNCK_LEN)
                else:
                    # print("??????????")
                    video_download_complete = self.players[re_download_video_id-ended_video_num].video_download(VIDEO_CHUNCK_LEN)
                # print("debug2:", download_video_id, chunk_id, self.players[re_download_video_id-ended_video_num].download_chunk_remain)

            if terminate_download == True:
                actual_download_size = self.network.cal_download_size(terminate_download_time, temp_throughput_arr, temp_duration_arr)
                reduce_wasted = chunk_size - actual_download_size
                self.reduce_wasted += reduce_wasted
                if re_download_video_id-ended_video_num == -1:
                    self.players[0].record_redundant_waste(-reduce_wasted)
                else:
                    self.players[re_download_video_id-ended_video_num].record_redundant_waste(-reduce_wasted)
                wasted -= reduce_wasted

            # if self.play_cnt == self.watch_num:  # if user leaves
            #     end_of_video = True
            # else:
            #     end_of_video = self.players[download_video_id].video_download(VIDEO_CHUNCK_LEN)

        # Sum up the bandwidth wastage
        # wasted_bytes += wasted
        self.total_waste += wasted
        if buffer < 0:
            rebuf = abs(buffer)

        past_throughput = self.network.past_throughput
        past_content, past_mod_watch_percent = self.get_past_video_info()
        queue_content, queue_buffer_size, queue_weighted_chunk_size = self.get_queue_video_info()
        # print(queue_content, queue_buffer_size, queue_weighted_chunk_size)
        if len(self.past_players) == 0:
            reward -= 0.5
        else:
            reward += self.past_players[0].reward
        reward = round(reward, 6)
        m_state = [past_throughput, past_content, queue_content, past_mod_watch_percent]
        s_state = [queue_buffer_size, queue_weighted_chunk_size, queue_content]

        return m_state, s_state, reward, done
