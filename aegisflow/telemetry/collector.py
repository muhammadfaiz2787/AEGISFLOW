import time
from collections import Counter

import psutil


class LiveNetworkCollector:

    def __init__(
        self,
        interval_seconds=2.0,
    ):
        self.interval_seconds = float(
            interval_seconds
        )

        self.previous_io = None
        self.previous_connections = None
        self.previous_time = None

    def _get_connections(self):

        connections = []

        try:
            raw_connections = psutil.net_connections(
                kind="inet"
            )

        except (
            psutil.AccessDenied,
            PermissionError,
        ):
            raw_connections = []

        for connection in raw_connections:

            remote_ip = None
            remote_port = None

            if connection.raddr:

                try:
                    remote_ip = connection.raddr.ip
                    remote_port = connection.raddr.port

                except AttributeError:

                    if len(connection.raddr) >= 2:
                        remote_ip = connection.raddr[0]
                        remote_port = connection.raddr[1]

            connections.append(
                {
                    "status": connection.status,
                    "remote_ip": remote_ip,
                    "remote_port": remote_port,
                }
            )

        return connections

    def _snapshot(self):

        io = psutil.net_io_counters()

        connections = self._get_connections()

        timestamp = time.time()

        return (
            io,
            connections,
            timestamp,
        )

    def collect_window(self):

        # ====================================================
        # INITIAL SNAPSHOT
        # ====================================================

        if self.previous_io is None:

            (
                self.previous_io,
                self.previous_connections,
                self.previous_time,
            ) = self._snapshot()

        # ====================================================
        # WAIT UNTIL WINDOW IS COMPLETE
        # ====================================================

        elapsed = (
            time.time()
            - self.previous_time
        )

        remaining = (
            self.interval_seconds
            - elapsed
        )

        if remaining > 0:
            time.sleep(
                remaining
            )

        # ====================================================
        # CURRENT SNAPSHOT
        # ====================================================

        (
            current_io,
            current_connections,
            current_time,
        ) = self._snapshot()

        delta_time = max(
            current_time
            - self.previous_time,
            1e-6,
        )

        # ====================================================
        # TRAFFIC DELTAS
        # ====================================================

        bytes_sent_delta = max(
            current_io.bytes_sent
            - self.previous_io.bytes_sent,
            0,
        )

        bytes_recv_delta = max(
            current_io.bytes_recv
            - self.previous_io.bytes_recv,
            0,
        )

        packets_sent_delta = max(
            current_io.packets_sent
            - self.previous_io.packets_sent,
            0,
        )

        packets_recv_delta = max(
            current_io.packets_recv
            - self.previous_io.packets_recv,
            0,
        )

        total_bytes_delta = (
            bytes_sent_delta
            + bytes_recv_delta
        )

        total_packets_delta = (
            packets_sent_delta
            + packets_recv_delta
        )

        packet_rate_raw = (
            total_packets_delta
            / delta_time
        )

        byte_rate_raw = (
            total_bytes_delta
            / delta_time
        )

        # ====================================================
        # CONNECTION METRICS
        # ====================================================

        active_connections = [
            item
            for item in current_connections
            if item["status"]
            in {
                "ESTABLISHED",
                "SYN_SENT",
                "SYN_RECV",
            }
        ]

        connection_count = len(
            active_connections
        )

        remote_ips = [
            item["remote_ip"]
            for item in active_connections
            if item["remote_ip"] is not None
        ]

        unique_destinations = len(
            set(remote_ips)
        )

        remote_ports = [
            item["remote_port"]
            for item in active_connections
            if item["remote_port"] is not None
        ]

        unique_ports = len(
            set(remote_ports)
        )

        syn_count = sum(
            1
            for item in current_connections
            if item["status"]
            in {
                "SYN_SENT",
                "SYN_RECV",
            }
        )

        established_count = sum(
            1
            for item in current_connections
            if item["status"]
            == "ESTABLISHED"
        )

        time_wait_count = sum(
            1
            for item in current_connections
            if item["status"]
            == "TIME_WAIT"
        )

        previous_remote_ips = {
            item["remote_ip"]
            for item in (
                self.previous_connections
                or []
            )
            if item["remote_ip"] is not None
        }

        current_remote_ips = set(
            remote_ips
        )

        new_destinations = len(
            current_remote_ips
            - previous_remote_ips
        )

        destination_counter = Counter(
            remote_ips
        )

        if destination_counter:

            most_common_destination_count = (
                destination_counter
                .most_common(1)[0][1]
            )

        else:
            most_common_destination_count = 0

        # ====================================================
        # OUTPUT
        # ====================================================

        metrics = {
            "timestamp":
                current_time,

            "window_seconds":
                delta_time,

            "bytes_sent_delta":
                bytes_sent_delta,

            "bytes_recv_delta":
                bytes_recv_delta,

            "packets_sent_delta":
                packets_sent_delta,

            "packets_recv_delta":
                packets_recv_delta,

            "packet_rate_raw":
                packet_rate_raw,

            "byte_rate_raw":
                byte_rate_raw,

            "connection_count":
                connection_count,

            "unique_destinations":
                unique_destinations,

            "unique_ports":
                unique_ports,

            "syn_count":
                syn_count,

            "established_count":
                established_count,

            "time_wait_count":
                time_wait_count,

            "new_destinations":
                new_destinations,

            "most_common_destination_count":
                most_common_destination_count,
        }

        # ====================================================
        # UPDATE INTERNAL STATE
        # ====================================================

        self.previous_io = current_io

        self.previous_connections = (
            current_connections
        )

        self.previous_time = (
            current_time
        )

        return metrics