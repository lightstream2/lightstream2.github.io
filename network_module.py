# shared by all players
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 0.08  # sec
PACKET_SIZE = 1500  # bytes
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0

class Network:
    def __init__(self, time, bandwidth):  # s & Mbps
        assert len(time) == len(bandwidth)

        self.time = [float(item) for item in time]
        self.bandwidth = [float(item) for item in bandwidth]

        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.time[self.mahimahi_ptr - 1]
        
        self.past_throughput = [0.1] * 10  # Mbps

    # calculate the download time of a certain block
    def network_simu(self,video_chunk_size):
        delay = 0.0  # in s
        video_chunk_counter_sent = 0  # in bytes
        temp_throughput_arr = []
        temp_duration_arr = []

        while True:  # download video chunk over mahimahi
            # print("video_chunk_size", video_chunk_size)
            throughput = self.bandwidth[self.mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE      # B/s
            duration = self.time[self.mahimahi_ptr] - self.last_mahimahi_time    # s
            # print("bandwidth: ", self.bandwidth[self.mahimahi_ptr], ", throughput: ", self.bandwidth[self.mahimahi_ptr] / BITS_IN_BYTE )
            # print("duration")
            self.past_throughput = [self.bandwidth[self.mahimahi_ptr]] + self.past_throughput[:-1]
            temp_throughput_arr.append(throughput)
            temp_duration_arr.append(duration)

            packet_payload = round(throughput * duration * PACKET_PAYLOAD_PORTION, 2)  # B
            # print("packet_payload", packet_payload)

            if video_chunk_counter_sent + packet_payload > video_chunk_size:  # B
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                # print("sending packet_payload: ", packet_payload, " in duration: ", fractional_time)
                delay += fractional_time    # s
                # print("debug1", fractional_time, self.last_mahimahi_time)
                self.last_mahimahi_time += fractional_time  # s
                # print("debug2", self.last_mahimahi_time)
                break

            video_chunk_counter_sent += packet_payload  # B
            # print("video_chunk_counter_sent", video_chunk_counter_sent)
            # print("sending packet ", video_chunk_counter_sent, ", packet_payload ", packet_payload, ', throughput ', throughput, ", duration ", duration)
            delay += duration  # s
            # print("sending packet_payload: ", packet_payload, " in duration: ", duration)
            self.last_mahimahi_time = self.time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.bandwidth):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay += LINK_RTT  # s
        delay = round(delay, 3)
        # print("delay: ", delay)
        return delay, temp_throughput_arr, temp_duration_arr
    
    # Calculate the download size in a given amount of time
    def cal_download_size(self, play_time, temp_throughput_arr, temp_duration_arr):
        play_time -= LINK_RTT
        temp = 0
        download_size = 0 # B
        while play_time > 0:
            throughput = temp_throughput_arr[temp]
            duration = temp_duration_arr[temp]

            if play_time >= duration:
                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION  # B
                download_size += packet_payload
                play_time -= duration
                temp += 1
            else:
                packet_payload = throughput * play_time * PACKET_PAYLOAD_PORTION  # B
                download_size += packet_payload
                play_time = 0

        return download_size
