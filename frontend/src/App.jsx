import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  Database,
  Gauge,
  Network,
  Play,
  Radio,
  Shield,
  ShieldCheck,
  Square,
  Wifi,
} from "lucide-react";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import MetricCard from "./components/MetricCard";
import StatusBadge from "./components/StatusBadge";
import ThreatChart from "./components/ThreatChart";
import TrafficChart from "./components/TrafficChart";


const API_BASE = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws/live";


function clamp01(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return 0;
  }

  return Math.max(
    0,
    Math.min(1, number)
  );
}


function percentage(value) {
  return `${(
    clamp01(value) * 100
  ).toFixed(1)}%`;
}


function formatThroughput(bytesPerSecond) {
  const value =
    Number(bytesPerSecond) || 0;

  if (value >= 1_000_000) {
    return `${(
      value / 1_000_000
    ).toFixed(2)} MB/s`;
  }

  if (value >= 1_000) {
    return `${(
      value / 1_000
    ).toFixed(1)} KB/s`;
  }

  return `${value.toFixed(0)} B/s`;
}


function getPolicyAccent(policy) {
  switch (
    String(policy).toUpperCase()
  ) {
    case "LOW":
      return "green";

    case "MEDIUM":
      return "blue";

    case "HIGH":
      return "yellow";

    case "CRITICAL":
      return "red";

    default:
      return "blue";
  }
}


function PerformanceRow({
  label,
  value,
}) {
  return (
    <div
      className="
        flex
        items-center
        justify-between
        gap-4
        border-b
        border-slate-800
        pb-3
      "
    >
      <span
        className="
          text-sm
          text-slate-500
        "
      >
        {label}
      </span>

      <span
        className="
          font-mono
          text-sm
          font-medium
          text-slate-200
        "
      >
        {value}
      </span>
    </div>
  );
}


function App() {
  const [
    monitoring,
    setMonitoring,
  ] = useState(false);

  const [
    latest,
    setLatest,
  ] = useState(null);

  const [
    sequence,
    setSequence,
  ] = useState(0);

  const [
    socketStatus,
    setSocketStatus,
  ] = useState("connecting");

  const [
    error,
    setError,
  ] = useState("");

  const [
    loading,
    setLoading,
  ] = useState(false);

  const [
    activeTab,
    setActiveTab,
  ] = useState("dashboard");

  const [
    history,
    setHistory,
  ] = useState([]);

  const [
    performance,
    setPerformance,
  ] = useState(null);

  const [
    config,
    setConfig,
  ] = useState({
    profile_name: "general",
    device_trust: 0.75,
    destination_trust: 0.75,
    latency_sensitivity: 0.5,
  });


  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(true);


  const metrics =
    latest?.metrics ?? {};

  const intelligence =
    latest?.intelligence ?? {};

  const temporalAnomaly =
    intelligence?.temporal_anomaly ?? {};

  const telemetry =
    latest?.telemetry ?? {};

  const decision =
    latest?.decision ?? {};

  const policyContext =
    latest?.policy_context ?? {};


  // =========================================================
  // INITIAL API LOAD + WEBSOCKET
  // =========================================================

  useEffect(() => {
    let cancelled = false;

    shouldReconnectRef.current = true;


    async function loadInitialData() {
      try {
        const [
          statusResponse,
          performanceResponse,
        ] = await Promise.all([
          fetch(
            `${API_BASE}/api/status`
          ),

          fetch(
            `${API_BASE}/api/model-performance`
          ),
        ]);

        if (!statusResponse.ok) {
          throw new Error(
            `Status API returned ${statusResponse.status}`
          );
        }

        if (!performanceResponse.ok) {
          throw new Error(
            `Performance API returned ${performanceResponse.status}`
          );
        }

        const [
          statusData,
          performanceData,
        ] = await Promise.all([
          statusResponse.json(),
          performanceResponse.json(),
        ]);

        if (cancelled) {
          return;
        }

        setMonitoring(
          Boolean(
            statusData.monitoring
          )
        );

        setSequence(
          statusData.sequence ?? 0
        );

        if (statusData.latest) {
          setLatest(
            statusData.latest
          );
        }

        if (statusData.config) {
          setConfig(
            statusData.config
          );
        }

        if (statusData.error) {
          setError(
            statusData.error
          );
        }

        setPerformance(
          performanceData
        );
      } catch (initialError) {
        if (cancelled) {
          return;
        }

        console.error(
          initialError
        );

        setError(
          "Cannot connect to AegisFlow backend."
        );
      }
    }


    function connectSocket() {
      if (cancelled) {
        return;
      }

      const oldSocket =
        socketRef.current;

      if (
        oldSocket &&
        (
          oldSocket.readyState ===
            WebSocket.OPEN ||
          oldSocket.readyState ===
            WebSocket.CONNECTING
        )
      ) {
        return;
      }

      setSocketStatus(
        "connecting"
      );

      const socket =
        new WebSocket(
          WS_URL
        );

      socketRef.current =
        socket;


      socket.onopen = () => {
        if (cancelled) {
          return;
        }

        setSocketStatus(
          "connected"
        );

        setError("");
      };


      socket.onmessage = event => {
        if (cancelled) {
          return;
        }

        try {
          const payload =
            JSON.parse(
              event.data
            );

          setMonitoring(
            Boolean(
              payload.monitoring
            )
          );

          setSequence(
            payload.sequence ?? 0
          );

          if (payload.error) {
            setError(
              payload.error
            );
          }

          if (!payload.data) {
            return;
          }

          setLatest(
            payload.data
          );

          const point = {
            sequence:
              payload.sequence ?? 0,

            threat:
              Number(
                payload.data
                  ?.intelligence
                  ?.threat_score
                ?? 0
              ),

            anomaly:
              Number(
                payload.data
                  ?.intelligence
                  ?.anomaly_score
                ?? 0
              ),

            packetRate:
              Number(
                payload.data
                  ?.metrics
                  ?.packet_rate_raw
                ?? 0
              ),

            throughput:
              Number(
                payload.data
                  ?.metrics
                  ?.byte_rate_raw
                ?? 0
              ),
          };

          setHistory(previous => {
            return [
              ...previous,
              point,
            ].slice(-40);
          });
        } catch (parseError) {
          console.error(
            "WebSocket parse error:",
            parseError
          );
        }
      };


      socket.onerror = () => {
        if (cancelled) {
          return;
        }

        setSocketStatus(
          "error"
        );
      };


      socket.onclose = () => {
        socketRef.current =
          null;

        if (
          cancelled ||
          !shouldReconnectRef.current
        ) {
          return;
        }

        setSocketStatus(
          "disconnected"
        );

        reconnectTimerRef.current =
          window.setTimeout(
            connectSocket,
            2000
          );
      };
    }


    void loadInitialData();
    connectSocket();


    return () => {
      cancelled = true;

      shouldReconnectRef.current =
        false;

      if (
        reconnectTimerRef.current
      ) {
        window.clearTimeout(
          reconnectTimerRef.current
        );
      }

      if (
        socketRef.current
      ) {
        socketRef.current.close();

        socketRef.current =
          null;
      }
    };
  }, []);


  // =========================================================
  // START MONITORING
  // =========================================================

  async function startMonitoring() {
    setLoading(true);
    setError("");

    try {
      const response =
        await fetch(
          `${API_BASE}/api/monitor/start`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(
                config
              ),
          }
        );

      const data =
        await response.json();

      if (!data.success) {
        throw new Error(
          data.message ??
          "Unable to start monitoring."
        );
      }

      setMonitoring(true);
      setHistory([]);
    } catch (startError) {
      console.error(
        startError
      );

      setError(
        startError.message
      );
    } finally {
      setLoading(false);
    }
  }


  // =========================================================
  // STOP MONITORING
  // =========================================================

  async function stopMonitoring() {
    setLoading(true);
    setError("");

    try {
      const response =
        await fetch(
          `${API_BASE}/api/monitor/stop`,
          {
            method: "POST",
          }
        );

      const data =
        await response.json();

      if (!data.success) {
        throw new Error(
          data.message ??
          "Unable to stop monitoring."
        );
      }

      setMonitoring(false);
    } catch (stopError) {
      console.error(
        stopError
      );

      setError(
        stopError.message
      );
    } finally {
      setLoading(false);
    }
  }


  // =========================================================
  // UPDATE SECURITY CONFIG
  // =========================================================

  async function updateConfig(
    nextConfig
  ) {
    setConfig(nextConfig);

    try {
      const response =
        await fetch(
          `${API_BASE}/api/config`,
          {
            method: "PUT",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(
                nextConfig
              ),
          }
        );

      if (!response.ok) {
        throw new Error(
          `Config update failed: ${response.status}`
        );
      }
    } catch (configError) {
      console.error(
        configError
      );

      setError(
        "Unable to update security context."
      );
    }
  }


  const anomalyDetected =
    Boolean(
      intelligence
        .predicted_anomaly
    );

  const temporalState =
    temporalAnomaly.state
    ?? "NORMAL";

  const behaviorLabel =
    temporalState ===
      "PERSISTENT_ANOMALY"
        ? "PERSISTENT"
        : temporalState;


  const policy =
    decision.policy ?? "—";


  const policyAccent =
    getPolicyAccent(
      policy
    );


  const chartHistory =
    useMemo(
      () => history,
      [history]
    );


  return (
    <div
      className="
        min-h-screen
        text-slate-200
      "
    >
      {/* HEADER */}

      <header
        className="
          sticky
          top-0
          z-50
          border-b
          border-slate-800/80
          bg-[#070b12]/90
          backdrop-blur-xl
        "
      >
        <div
          className="
            mx-auto
            flex
            max-w-[1600px]
            items-center
            justify-between
            px-5
            py-4
            lg:px-8
          "
        >
          <div
            className="
              flex
              items-center
              gap-3
            "
          >
            <div
              className="
                flex
                h-10
                w-10
                items-center
                justify-center
                rounded-xl
                border
                border-blue-500/20
                bg-blue-500/10
              "
            >
              <ShieldCheck
                className="
                  h-5
                  w-5
                  text-blue-400
                "
              />
            </div>

            <div>
              <h1
                className="
                  text-lg
                  font-semibold
                  tracking-tight
                  text-white
                "
              >
                AegisFlow
              </h1>

              <p
                className="
                  hidden
                  text-xs
                  text-slate-500
                  sm:block
                "
              >
                Adaptive Security Intelligence
              </p>
            </div>
          </div>


          <div
            className="
              flex
              items-center
              gap-3
            "
          >
            <StatusBadge
              active={
                socketStatus ===
                "connected"
              }
              activeText="API CONNECTED"
              inactiveText="API OFFLINE"
            />

            <StatusBadge
              active={monitoring}
              activeText="MONITORING"
              inactiveText="STOPPED"
            />
          </div>
        </div>
      </header>


      {/* MAIN */}

      <main
        className="
          mx-auto
          max-w-[1600px]
          px-5
          py-8
          lg:px-8
        "
      >
        <div
          className="
            mb-8
            flex
            flex-col
            gap-5
            xl:flex-row
            xl:items-end
            xl:justify-between
          "
        >
          <div>
            <p
              className="
                mb-2
                text-xs
                font-semibold
                uppercase
                tracking-[0.22em]
                text-blue-400
              "
            >
              Security Operations
            </p>

            <h2
              className="
                text-3xl
                font-semibold
                tracking-tight
                text-white
              "
            >
              Network Intelligence Dashboard
            </h2>

            <p
              className="
                mt-2
                max-w-2xl
                text-sm
                text-slate-400
              "
            >
              Real-time network behavior,
              experimental threat intelligence,
              and adaptive security policy.
            </p>
          </div>


          <div
            className="
              flex
              gap-3
            "
          >
            <button
              type="button"
              onClick={() =>
                setActiveTab(
                  "dashboard"
                )
              }
              className={`
                rounded-xl
                border
                px-4
                py-2.5
                text-sm
                ${
                  activeTab ===
                  "dashboard"
                    ? `
                      border-blue-500/30
                      bg-blue-500/10
                      text-blue-300
                    `
                    : `
                      border-slate-800
                      bg-slate-900
                      text-slate-400
                    `
                }
              `}
            >
              Live Dashboard
            </button>

            <button
              type="button"
              onClick={() =>
                setActiveTab(
                  "performance"
                )
              }
              className={`
                rounded-xl
                border
                px-4
                py-2.5
                text-sm
                ${
                  activeTab ===
                  "performance"
                    ? `
                      border-blue-500/30
                      bg-blue-500/10
                      text-blue-300
                    `
                    : `
                      border-slate-800
                      bg-slate-900
                      text-slate-400
                    `
                }
              `}
            >
              Model Performance
            </button>
          </div>
        </div>


        {error && (
          <div
            className="
              mb-6
              flex
              gap-3
              rounded-xl
              border
              border-red-500/20
              bg-red-500/10
              p-4
              text-sm
              text-red-300
            "
          >
            <AlertTriangle
              className="
                h-5
                w-5
              "
            />

            {error}
          </div>
        )}


        {activeTab ===
          "dashboard" && (
          <>
            {/* MAIN METRICS */}

            <section
              className="
                mb-6
                grid
                grid-cols-1
                gap-4
                md:grid-cols-2
                xl:grid-cols-4
              "
            >
              <MetricCard
                title="Threat Score"
                value={
                  percentage(
                    intelligence
                      .threat_score
                  )
                }
                subtitle="Experimental real-network estimate"
                accent="blue"
                icon={
                  <BrainCircuit
                    className="h-5 w-5"
                  />
                }
              />

              <MetricCard
                title="Network Behavior"
                value={behaviorLabel}
                subtitle={
                  temporalState === "NORMAL"
                    ? "Network behavior within baseline"

                    : temporalState === "OBSERVING"
                      ? "Unusual activity observed"

                      : temporalState === "WARNING"
                        ? "Repeated baseline deviation"

                        : "Persistent network anomaly"
                }
                accent={
                  temporalState ===
                  "PERSISTENT_ANOMALY"
                    ? "red"
                    : temporalState ===
                      "WARNING"
                      ? "yellow"
                      : temporalState ===
                        "OBSERVING"
                        ? "purple"
                        : "green"
                }
                icon={
                  anomalyDetected
                    ? (
                      <AlertTriangle
                        className="h-5 w-5"
                      />
                    )
                    : (
                      <ShieldCheck
                        className="h-5 w-5"
                      />
                    )
                }
              />

              <MetricCard
                title="Active Policy"
                value={policy}
                subtitle="Context-aware security level"
                accent={policyAccent}
                icon={
                  <Shield
                    className="h-5 w-5"
                  />
                }
              />

              <MetricCard
                title="Packet Rate"
                value={`${Number(
                  metrics.packet_rate_raw ??
                  0
                ).toFixed(1)} pkt/s`}
                subtitle={
                  formatThroughput(
                    metrics.byte_rate_raw
                  )
                }
                accent="purple"
                icon={
                  <Activity
                    className="h-5 w-5"
                  />
                }
              />
            </section>


            {/* CHARTS */}

            <section
              className="
                mb-6
                grid
                grid-cols-1
                gap-6
                xl:grid-cols-2
              "
            >
              <div
                className="
                  rounded-2xl
                  border
                  border-slate-800
                  bg-slate-950/70
                  p-5
                "
              >
                <div
                  className="
                    mb-5
                    flex
                    items-center
                    justify-between
                  "
                >
                  <div>
                    <h3
                      className="
                        font-semibold
                        text-white
                      "
                    >
                      Intelligence History
                    </h3>

                    <p
                      className="
                        mt-1
                        text-xs
                        text-slate-500
                      "
                    >
                      Threat and anomaly score
                    </p>
                  </div>

                  <Radio
                    className="
                      h-5
                      w-5
                      text-blue-400
                    "
                  />
                </div>

                <ThreatChart
                  data={
                    chartHistory
                  }
                />
              </div>


              <div
                className="
                  rounded-2xl
                  border
                  border-slate-800
                  bg-slate-950/70
                  p-5
                "
              >
                <div
                  className="
                    mb-5
                    flex
                    items-center
                    justify-between
                  "
                >
                  <div>
                    <h3
                      className="
                        font-semibold
                        text-white
                      "
                    >
                      Network Traffic
                    </h3>

                    <p
                      className="
                        mt-1
                        text-xs
                        text-slate-500
                      "
                    >
                      Packet rate per monitoring window
                    </p>
                  </div>

                  <Network
                    className="
                      h-5
                      w-5
                      text-emerald-400
                    "
                  />
                </div>

                <TrafficChart
                  data={
                    chartHistory
                  }
                />
              </div>
            </section>


            {/* LOWER CONTENT */}

            <section
              className="
                grid
                grid-cols-1
                gap-6
                xl:grid-cols-[1.2fr_0.8fr]
              "
            >
              <div
                className="
                  rounded-2xl
                  border
                  border-slate-800
                  bg-slate-950/70
                  p-5
                "
              >
                <div
                  className="
                    mb-5
                    flex
                    items-center
                    gap-2
                  "
                >
                  <Wifi
                    className="
                      h-5
                      w-5
                      text-blue-400
                    "
                  />

                  <h3
                    className="
                      font-semibold
                      text-white
                    "
                  >
                    Live Telemetry
                  </h3>
                </div>


                <div
                  className="
                    grid
                    grid-cols-2
                    gap-3
                  "
                >
                  <TelemetryCard
                    label="Connections"
                    value={
                      metrics.connection_count ??
                      0
                    }
                  />

                  <TelemetryCard
                    label="Destinations"
                    value={
                      metrics.unique_destinations ??
                      0
                    }
                  />

                  <TelemetryCard
                    label="Anomaly Score"
                    value={
                      percentage(
                        intelligence
                          .anomaly_score
                      )
                    }
                  />

                  <TelemetryCard
                    label="Sequence"
                    value={
                      sequence
                    }
                  />

                  <TelemetryCard
                    label="Reconstruction Error"
                    value={
                      Number(
                        intelligence
                          .reconstruction_error ??
                        0
                      ).toFixed(6)
                    }
                  />

                  <TelemetryCard
                    label="Anomaly Threshold"
                    value={
                      Number(
                        intelligence
                          .anomaly_threshold ??
                        0
                      ).toFixed(6)
                    }
                  />

                  <TelemetryCard
                    label="Temporal State"
                    value={
                      temporalState
                    }
                  />

                  <TelemetryCard
                    label="Anomalous Windows"
                    value={
                      `${
                        temporalAnomaly
                          .anomaly_count
                        ?? 0
                      } / ${
                        temporalAnomaly
                          .observed_windows
                        ?? 0
                      }`
                    }
                  />

                  <div
                    className="
                      col-span-2
                      rounded-xl
                      border
                      border-slate-800
                      bg-slate-900/40
                      p-4
                    "
                  >
                    <div
                      className="
                        mb-2
                        flex
                        justify-between
                        text-xs
                      "
                    >
                      <span
                        className="
                          text-slate-500
                        "
                      >
                        Temporal anomaly ratio
                      </span>

                      <span
                        className="
                          text-slate-300
                        "
                      >
                        {percentage(
                          temporalAnomaly
                            .anomaly_ratio
                        )}
                      </span>
                    </div>

                    <div
                      className="
                        h-2
                        overflow-hidden
                        rounded-full
                        bg-slate-800
                      "
                    >
                      <div
                        className="
                          h-full
                          rounded-full
                          bg-amber-400
                          transition-all
                          duration-300
                        "
                        style={{
                          width:
                            percentage(
                              temporalAnomaly
                                .anomaly_ratio
                            ),
                        }}
                      />
                    </div>
                  </div>
                </div>


                <div
                  className="
                    mt-6
                    border-t
                    border-slate-800
                    pt-5
                  "
                >
                  <p
                    className="
                      mb-3
                      text-xs
                      font-semibold
                      uppercase
                      tracking-wider
                      text-slate-500
                    "
                  >
                    Behavioral Features
                  </p>


                  <div
                    className="
                      grid
                      grid-cols-1
                      gap-x-8
                      md:grid-cols-2
                    "
                  >
                    {Object.entries(
                      telemetry
                    ).map(
                      ([
                        name,
                        value,
                      ]) => (
                        <div
                          key={name}
                          className="
                            flex
                            justify-between
                            gap-4
                            border-b
                            border-slate-800
                            py-2
                            text-xs
                          "
                        >
                          <span
                            className="
                              truncate
                              text-slate-500
                            "
                          >
                            {name}
                          </span>

                          <span
                            className="
                              font-mono
                              text-slate-300
                            "
                          >
                            {Number(
                              value
                            ).toFixed(4)}
                          </span>
                        </div>
                      )
                    )}
                  </div>
                </div>
              </div>


              <div
                className="
                  space-y-6
                "
              >
                {/* SECURITY CONTEXT */}

                <div
                  className="
                    rounded-2xl
                    border
                    border-slate-800
                    bg-slate-950/70
                    p-5
                  "
                >
                  <div
                    className="
                      mb-5
                      flex
                      items-center
                      gap-2
                    "
                  >
                    <Database
                      className="
                        h-5
                        w-5
                        text-violet-400
                      "
                    />

                    <h3
                      className="
                        font-semibold
                        text-white
                      "
                    >
                      Security Context
                    </h3>
                  </div>


                  <label
                    className="
                      mb-2
                      block
                      text-xs
                      text-slate-500
                    "
                  >
                    Data Profile
                  </label>


                  <select
                    value={
                      config.profile_name
                    }
                    onChange={
                      event => {
                        const next = {
                          ...config,

                          profile_name:
                            event.target.value,
                        };

                        void updateConfig(
                          next
                        );
                      }
                    }
                    className="
                      mb-6
                      w-full
                      rounded-xl
                      border
                      border-slate-700
                      bg-slate-900
                      px-3
                      py-2.5
                      text-sm
                    "
                  >
                    <option value="general">
                      General
                    </option>

                    <option value="personal">
                      Personal
                    </option>

                    <option value="financial">
                      Financial
                    </option>

                    <option value="medical">
                      Medical
                    </option>

                    <option value="iot">
                      IoT
                    </option>
                  </select>


                  {[
                    {
                      label:
                        "Device Trust",
                      key:
                        "device_trust",
                    },

                    {
                      label:
                        "Destination Trust",
                      key:
                        "destination_trust",
                    },

                    {
                      label:
                        "Latency Priority",
                      key:
                        "latency_sensitivity",
                    },
                  ].map(item => (
                    <div
                      key={item.key}
                      className="mb-5"
                    >
                      <div
                        className="
                          mb-2
                          flex
                          justify-between
                          text-xs
                        "
                      >
                        <span
                          className="
                            text-slate-400
                          "
                        >
                          {item.label}
                        </span>

                        <span
                          className="
                            text-white
                          "
                        >
                          {percentage(
                            config[
                              item.key
                            ]
                          )}
                        </span>
                      </div>


                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={
                          config[
                            item.key
                          ]
                        }
                        onChange={
                          event => {
                            const next = {
                              ...config,

                              [item.key]:
                                Number(
                                  event
                                    .target
                                    .value
                                ),
                            };

                            void updateConfig(
                              next
                            );
                          }
                        }
                        className="
                          w-full
                          accent-blue-500
                        "
                      />
                    </div>
                  ))}


                  <div
                    className="
                      rounded-xl
                      border
                      border-slate-800
                      bg-slate-900/40
                      p-4
                    "
                  >
                    <p
                      className="
                        text-xs
                        text-slate-500
                      "
                    >
                      Current Network Threat
                    </p>

                    <p
                      className="
                        mt-2
                        text-xl
                        font-semibold
                        text-white
                      "
                    >
                      {percentage(
                        policyContext
                          .network_threat
                      )}
                    </p>
                  </div>
                </div>


                {/* CONTROL */}

                <div
                  className="
                    rounded-2xl
                    border
                    border-slate-800
                    bg-slate-950/70
                    p-5
                  "
                >
                  <h3
                    className="
                      mb-4
                      font-semibold
                      text-white
                    "
                  >
                    Monitoring Control
                  </h3>


                  {!monitoring ? (
                    <button
                      type="button"
                      onClick={
                        startMonitoring
                      }
                      disabled={loading}
                      className="
                        flex
                        w-full
                        items-center
                        justify-center
                        gap-2
                        rounded-xl
                        bg-blue-600
                        px-4
                        py-3
                        font-semibold
                        text-white
                        hover:bg-blue-500
                        disabled:opacity-50
                      "
                    >
                      <Play
                        className="h-4 w-4"
                      />

                      Start Monitoring
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={
                        stopMonitoring
                      }
                      disabled={loading}
                      className="
                        flex
                        w-full
                        items-center
                        justify-center
                        gap-2
                        rounded-xl
                        border
                        border-red-500/30
                        bg-red-500/10
                        px-4
                        py-3
                        font-semibold
                        text-red-300
                        disabled:opacity-50
                      "
                    >
                      <Square
                        className="h-4 w-4"
                      />

                      Stop Monitoring
                    </button>
                  )}


                  <div
                    className="
                      mt-4
                      flex
                      justify-between
                      text-xs
                      text-slate-500
                    "
                  >
                    <span>
                      WebSocket
                    </span>

                    <span>
                      {socketStatus}
                    </span>
                  </div>
                </div>
              </div>
            </section>
          </>
        )}


        {/* MODEL PERFORMANCE */}

        {activeTab ===
          "performance" && (
          <section>
            <div
              className="
                mb-6
                rounded-2xl
                border
                border-amber-500/20
                bg-amber-500/5
                p-5
              "
            >
              <div
                className="
                  flex
                  gap-3
                "
              >
                <AlertTriangle
                  className="
                    h-5
                    w-5
                    text-amber-400
                  "
                />

                <div>
                  <h3
                    className="
                      font-medium
                      text-amber-300
                    "
                  >
                    Evaluation Scope
                  </h3>

                  <p
                    className="
                      mt-1
                      text-sm
                      text-slate-400
                    "
                  >
                    Metrics below come from
                    the final held-out
                    synthetic test and must
                    not be interpreted as
                    real-world network
                    accuracy.
                  </p>
                </div>
              </div>
            </div>


            <div
              className="
                grid
                grid-cols-1
                gap-6
                xl:grid-cols-3
              "
            >
              <PerformancePanel
                icon={
                  <Shield
                    className="
                      h-6
                      w-6
                      text-blue-400
                    "
                  />
                }
                title="Policy Engine"
              >
                <PerformanceRow
                  label="Oracle Agreement"
                  value={`${
                    performance
                      ?.policy
                      ?.oracle_agreement_pct
                      ?.toFixed(2)
                    ?? "—"
                  }%`}
                />

                <PerformanceRow
                  label="Mean Reward"
                  value={
                    performance
                      ?.policy
                      ?.mean_reward
                      ?.toFixed(4)
                    ?? "—"
                  }
                />

                <PerformanceRow
                  label="Mean Regret"
                  value={
                    performance
                      ?.policy
                      ?.mean_regret
                      ?.toFixed(6)
                    ?? "—"
                  }
                />
              </PerformancePanel>


              <PerformancePanel
                icon={
                  <BrainCircuit
                    className="
                      h-6
                      w-6
                      text-violet-400
                    "
                  />
                }
                title="Threat Estimator"
              >
                <PerformanceRow
                  label="R²"
                  value={
                    performance
                      ?.threat_estimator
                      ?.r2
                      ?.toFixed(4)
                    ?? "—"
                  }
                />

                <PerformanceRow
                  label="Correlation"
                  value={
                    performance
                      ?.threat_estimator
                      ?.correlation
                      ?.toFixed(4)
                    ?? "—"
                  }
                />

                <PerformanceRow
                  label="MAE"
                  value={
                    performance
                      ?.threat_estimator
                      ?.mae
                      ?.toFixed(4)
                    ?? "—"
                  }
                />

                <PerformanceRow
                  label="RMSE"
                  value={
                    performance
                      ?.threat_estimator
                      ?.rmse
                      ?.toFixed(4)
                    ?? "—"
                  }
                />
              </PerformancePanel>


              <PerformancePanel
                icon={
                  <Gauge
                    className="
                      h-6
                      w-6
                      text-emerald-400
                    "
                  />
                }
                title="Synthetic Anomaly Model"
              >
                <PerformanceRow
                  label="ROC-AUC"
                  value={
                    performance
                      ?.synthetic_anomaly_detector
                      ?.roc_auc
                      ?.toFixed(4)
                    ?? "—"
                  }
                />

                <PerformanceRow
                  label="PR-AUC"
                  value={
                    performance
                      ?.synthetic_anomaly_detector
                      ?.pr_auc
                      ?.toFixed(4)
                    ?? "—"
                  }
                />

                <PerformanceRow
                  label="F1"
                  value={
                    performance
                      ?.synthetic_anomaly_detector
                      ?.f1
                      ?.toFixed(4)
                    ?? "—"
                  }
                />
              </PerformancePanel>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}


function TelemetryCard({
  label,
  value,
}) {
  return (
    <div
      className="
        rounded-xl
        border
        border-slate-800
        bg-slate-900/40
        p-4
      "
    >
      <p
        className="
          text-xs
          text-slate-500
        "
      >
        {label}
      </p>

      <p
        className="
          mt-2
          font-medium
          text-slate-200
        "
      >
        {value}
      </p>
    </div>
  );
}


function PerformancePanel({
  icon,
  title,
  children,
}) {
  return (
    <div
      className="
        rounded-2xl
        border
        border-slate-800
        bg-slate-950/70
        p-6
      "
    >
      <div className="mb-5">
        {icon}
      </div>

      <h3
        className="
          text-lg
          font-semibold
          text-white
        "
      >
        {title}
      </h3>

      <div
        className="
          mt-6
          space-y-4
        "
      >
        {children}
      </div>
    </div>
  );
}


export default App;