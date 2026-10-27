/*
 * low-altitude-manet-ai3.cc
 * ============================================================================
 * 低空无人机集群 多频段 MANET 组网与环境感知仿真实例 (AI3增强版)
 * ============================================================================
 * 适用 NS-3 版本 : 3.48
 * 放置位置      : ns-allinone-3.48/ns-3.48/scratch/low-altitude-manet-ai3.cc
 * 编译运行      :
 *   $ ./ns3 build
 *   $ ./ns3 run "scratch/low-altitude-manet-ai3 --scenario=0 --simTime=250"
 *
 * 参数说明:
 *   --scenario : 0=正常LOS, 1=建筑物遮挡NLOS, 2=同频干扰, 3=降雨衰减
 *   --simTime  : 仿真时长(秒), 默认 250s
 *
 * 本版本改动 (契合调研3 DeepMIMO/RadioMapSeer/WAIR-D 数据格式):
 *   - 新增 ai3-trace-deepmimo.csv: 契合 DeepMIMO 的位置+路径损耗+LoS/NLoS 数据
 *   - 新增 ai3-meta-radiomapseer.csv: 契合 RadioMapSeer/WAIR-D 的场景元数据
 *   - 场景1增加虚拟建筑几何遮挡判定，生成真实NLoS标签
 *   - 场景3增加ITU-R P.838雨衰模型近似
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/aodv-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/wifi-phy-common.h"

#include <fstream>
#include <iomanip>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("LowAltitudeManetAi3");

// ============================================================================
// 全局日志文件流
// ============================================================================
std::ofstream g_rssiLog;
std::ofstream g_envLog;

// ============================================================================
// AI3-Trace 输出流  (契合调研3 WAIR-D / RadioMapSeer / DeepMIMO 格式)
//   每行包含: 收发位置、频点、路径损耗、LoS/NLoS标签、场景标记、时戳
// ============================================================================
std::ofstream g_ai3Trace;      // AI3-Trace 主日志 (兼容DeepMIMO/WAIR-D/RadioMapSeer)
std::ofstream g_ai3Meta;       // 元数据: 场景标签、障碍物地图索引
std::ofstream g_carlaOut;      // CARLA轨迹输出 (供后续AI训练复用)

// ============================================================================
// RSSI / SINR Trace 回调
// ns-3.48 中 MonitorSnifferRx trace 签名增加了一个 uint16_t staId 参数
// ============================================================================
static void
PhyRxTrace(std::string context, Ptr<const Packet> /* packet */,
           uint16_t channelFreqMhz, WifiTxVector txVector,
           MpduInfo /* mpduInfo */, SignalNoiseDbm signalNoise,
           uint16_t /* staId */)
{
    // 从 context 字符串解析接收节点ID
    size_t pos = context.find("NodeList/");
    if (pos == std::string::npos) return;
    pos += 9;
    size_t end = context.find("/", pos);
    uint32_t rxNodeId = static_cast<uint32_t>(std::stoi(context.substr(pos, end - pos)));

    double rssi  = signalNoise.signal;
    double noise = signalNoise.noise;

    // 使用模式唯一名称替代 MCS 值, 避免 GetMcsValue() 断言崩溃
    std::string modeName = txVector.GetMode().GetUniqueName();

    g_rssiLog << std::fixed << std::setprecision(3)
              << Simulator::Now().GetSeconds() << ","
              << rxNodeId       << ","
              << channelFreqMhz << ","
              << modeName       << ","
              << rssi           << ","
              << noise          << "\n";
    g_rssiLog.flush();
}

// ============================================================================
// 周期性环境状态快照
// ============================================================================
static void
LogEnvironmentSnapshot(NodeContainer nodes)
{
    double now = Simulator::Now().GetSeconds();

    for (uint32_t i = 0; i < nodes.GetN(); ++i) {
        for (uint32_t j = i + 1; j < nodes.GetN(); ++j) {
            Ptr<MobilityModel> mobi = nodes.Get(i)->GetObject<MobilityModel>();
            Ptr<MobilityModel> mobj = nodes.Get(j)->GetObject<MobilityModel>();
            if (!mobi || !mobj) continue;

            double dist = mobi->GetDistanceFrom(mobj);
            double pl = (dist < 1.0) ? 40.0 : 40.0 + 20.0 * std::log10(dist);

            g_envLog << std::fixed << std::setprecision(3)
                     << now << ","
                     << i   << ","
                     << j   << ","
                     << dist << ","
                     << pl   << "\n";
        }
    }
    g_envLog.flush();

    Simulator::Schedule(Seconds(1.0), &LogEnvironmentSnapshot, nodes);
}

// ============================================================================
// AI3-Trace 周期性快照
// 契合调研3中 DeepMIMO / RadioMapSeer / WAIR-D 数据格式:
//   DeepMIMO  : 发射机/接收机位置、信道功率、LoS/NLoS指示符
//   RadioMapSeer: 城市场景路径损耗地图数据
//   WAIR-D    : 稀疏射频采样接收功率与障碍物反演数据
//
// 输出列    : Scenario,Time_s,TxNode,Tx_X,Tx_Y,Tx_Z,RxNode,Rx_X,Rx_Y,Rx_Z,
//             Distance_m,PathLoss_dB,LoS_Status,ScenarioTag,Freq_MHz
// ============================================================================
static void
Ai3TraceSnapshot(NodeContainer nodes, uint32_t scenario)
{
    double now = Simulator::Now().GetSeconds();

    // 场景1虚拟建筑区域 (用于NLoS判定)
    // 建筑占据中心区域: 200<x<300, 200<y<300, 0<z<45
    const double buildXMin = 200.0, buildXMax = 300.0;
    const double buildYMin = 200.0, buildYMax = 300.0;
    const double buildZMax = 45.0;

    for (uint32_t i = 0; i < nodes.GetN(); ++i) {
        for (uint32_t j = i + 1; j < nodes.GetN(); ++j) {
            Ptr<MobilityModel> mobi = nodes.Get(i)->GetObject<MobilityModel>();
            Ptr<MobilityModel> mobj = nodes.Get(j)->GetObject<MobilityModel>();
            if (!mobi || !mobj) continue;

            Vector pi = mobi->GetPosition();
            Vector pj = mobj->GetPosition();
            double dist = mobi->GetDistanceFrom(mobj);

            // 基础路径损耗 (自由空间参考)
            double pl = (dist < 1.0) ? 40.0 : 40.0 + 20.0 * std::log10(dist);

            // LoS/NLoS 状态与场景附加损耗
            std::string losTag = "LoS";
            std::string scenarioTag;
            double plExtra = 0.0;

            switch (scenario) {
                case 0:
                    scenarioTag = "LOS";
                    break;

                case 1: {
                    scenarioTag = "BUILDING_NLOS";
                    // 简化的线段-建筑相交判定
                    // 当中心落入建筑区域且两端高度均低于建筑时判定为NLoS
                    bool blocked = false;
                    double midX = (pi.x + pj.x) * 0.5;
                    double midY = (pi.y + pj.y) * 0.5;
                    double midZ = (pi.z + pj.z) * 0.5;
                    if (midX > buildXMin && midX < buildXMax &&
                        midY > buildYMin && midY < buildYMax &&
                        midZ < buildZMax) {
                        if (pi.z < buildZMax && pj.z < buildZMax) {
                            blocked = true;
                        }
                    }
                    // 补充: 若任一端点在建筑内也视为NLoS
                    if (!blocked) {
                        auto inBuilding = [&](const Vector& p) -> bool {
                            return (p.x > buildXMin && p.x < buildXMax &&
                                    p.y > buildYMin && p.y < buildYMax &&
                                    p.z < buildZMax);
                        };
                        if (inBuilding(pi) || inBuilding(pj)) blocked = true;
                    }
                    if (blocked) {
                        losTag = "NLoS";
                        // 建筑穿透损耗+额外绕射损耗 (契合调研3建筑物遮挡模型)
                        plExtra = 15.0 + 5.0 * std::log10(dist + 1.0);
                    }
                    break;
                }

                case 2:
                    scenarioTag = "INTERFERENCE";
                    break;

                case 3:
                    scenarioTag = "RAIN";
                    // ITU-R P.838 雨衰近似 (2.4GHz, 25mm/h降雨率)
                    // 2.4GHz雨衰系数约 0.02 dB/km, 极轻微
                    plExtra = 0.02 * (dist / 1000.0);
                    break;
            }

            pl += plExtra;

            // 契合 DeepMIMO 格式: 场景/发射机/接收机位置/路径损耗/LoS状态
            g_ai3Trace << std::fixed << std::setprecision(3)
                       << scenario << ","
                       << now << ","
                       << i << "," << pi.x << "," << pi.y << "," << pi.z << ","
                       << j << "," << pj.x << "," << pj.y << "," << pj.z << ","
                       << dist << ","
                       << pl << ","
                       << losTag << ","
                       << scenarioTag << ","
                       << "2400"             // 2.4 GHz中心频点 (MHz)
                       << "\n";
        }
    }
    g_ai3Trace.flush();

    Simulator::Schedule(Seconds(1.0), &Ai3TraceSnapshot, nodes, scenario);
}

// ============================================================================
// CARLA 轨迹输出 (契合调研3中 CARLA 仿真器轨迹数据集格式)
// ============================================================================
static void
LogCarlaTrajectory(NodeContainer nodes)
{
    double now = Simulator::Now().GetSeconds();

    for (uint32_t i = 0; i < nodes.GetN(); ++i) {
        Ptr<MobilityModel> mob = nodes.Get(i)->GetObject<MobilityModel>();
        if (!mob) continue;
        Vector pos = mob->GetPosition();
        Vector vel = mob->GetVelocity();
        g_carlaOut << std::fixed << std::setprecision(3)
                   << now << ","
                   << i << ","
                   << pos.x << "," << pos.y << "," << pos.z << ","
                   << vel.x << "," << vel.y << "," << vel.z << "\n";
    }
    g_carlaOut.flush();

    Simulator::Schedule(Seconds(0.1), &LogCarlaTrajectory, nodes);
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char *argv[])
{
    uint32_t scenario = 0;
    double   simTime   = 250.0;
    bool     enablePcap = false;
    bool     verbose    = true;

    CommandLine cmd(__FILE__);
    cmd.AddValue("scenario", "环境场景类型: 0=正常LOS, 1=建筑物遮挡NLOS, 2=同频干扰, 3=降雨衰减", scenario);
    cmd.AddValue("simTime",  "仿真时长(秒)", simTime);
    cmd.AddValue("pcap",     "是否启用PCAP抓包(true/false)", enablePcap);
    cmd.AddValue("verbose",  "是否启用AODV详细日志(true/false)", verbose);
    cmd.Parse(argc, argv);

    if (scenario > 3) {
        NS_FATAL_ERROR("--scenario 必须是 0~3 之间的整数");
    }

    const char* scenarioNames[] = {"正常LOS", "建筑物遮挡NLOS", "同频干扰", "降雨衰减"};
    std::cout << "\n========================================" << std::endl;
    std::cout << " 低空无人机 MANET 组网仿真 (AI3增强版)" << std::endl;
    std::cout << " 场景: " << scenarioNames[scenario] << std::endl;
    std::cout << " 时长: " << simTime << "s" << std::endl;
    std::cout << "========================================\n" << std::endl;

    if (verbose) {
        LogComponentEnable("LowAltitudeManetAi3", LOG_LEVEL_INFO);
        LogComponentEnable("AodvRoutingProtocol", LOG_LEVEL_WARN);
    }

    // ------------------------------------------------------------------------
    // 创建节点
    // ------------------------------------------------------------------------
    const uint32_t nUav = 4;
    const uint32_t nGs  = 2;
    NodeContainer uavNodes;
    uavNodes.Create(nUav);
    NodeContainer gsNodes;
    gsNodes.Create(nGs);

    NodeContainer allNodes;
    allNodes.Add(uavNodes);
    allNodes.Add(gsNodes);

    NodeContainer interfererNodes;
    if (scenario == 2) {
        interfererNodes.Create(1);
        allNodes.Add(interfererNodes);
    }

    // ------------------------------------------------------------------------
    // 移动模型配置
    // ns-3.48 中 Waypoint 构造函数参数顺序为 (Time, Vector)
    // ------------------------------------------------------------------------
    MobilityHelper mobUav1;
    mobUav1.SetMobilityModel("ns3::WaypointMobilityModel");
    mobUav1.Install(uavNodes.Get(0));
    Ptr<WaypointMobilityModel> wp1 = DynamicCast<WaypointMobilityModel>(
        uavNodes.Get(0)->GetObject<MobilityModel>());
    wp1->AddWaypoint(Waypoint(Seconds(0),    Vector(0,    0,   50)));
    wp1->AddWaypoint(Waypoint(Seconds(50),   Vector(500,  0,   50)));
    wp1->AddWaypoint(Waypoint(Seconds(100),  Vector(500,  500, 50)));
    wp1->AddWaypoint(Waypoint(Seconds(150),  Vector(0,    500, 50)));
    wp1->AddWaypoint(Waypoint(Seconds(200),  Vector(0,    0,   50)));
    wp1->AddWaypoint(Waypoint(Seconds(simTime), Vector(0, 0,    50)));

    MobilityHelper mobUav2;
    mobUav2.SetMobilityModel("ns3::WaypointMobilityModel");
    mobUav2.Install(uavNodes.Get(1));
    Ptr<WaypointMobilityModel> wp2 = DynamicCast<WaypointMobilityModel>(
        uavNodes.Get(1)->GetObject<MobilityModel>());
    wp2->AddWaypoint(Waypoint(Seconds(0),    Vector(250, 250, 30)));
    wp2->AddWaypoint(Waypoint(Seconds(20),   Vector(300, 250, 30)));
    wp2->AddWaypoint(Waypoint(Seconds(40),   Vector(300, 300, 30)));
    wp2->AddWaypoint(Waypoint(Seconds(60),   Vector(250, 300, 30)));
    wp2->AddWaypoint(Waypoint(Seconds(80),   Vector(200, 300, 30)));
    wp2->AddWaypoint(Waypoint(Seconds(100),  Vector(200, 250, 30)));
    wp2->AddWaypoint(Waypoint(Seconds(120),  Vector(200, 200, 30)));
    wp2->AddWaypoint(Waypoint(Seconds(140),  Vector(250, 200, 30)));
    wp2->AddWaypoint(Waypoint(Seconds(160),  Vector(300, 200, 30)));
    wp2->AddWaypoint(Waypoint(Seconds(180),  Vector(300, 250, 30)));
    wp2->AddWaypoint(Waypoint(Seconds(200),  Vector(250, 250, 30)));
    wp2->AddWaypoint(Waypoint(Seconds(simTime), Vector(250, 250, 30)));

    MobilityHelper mobUav3;
    mobUav3.SetMobilityModel("ns3::WaypointMobilityModel");
    mobUav3.Install(uavNodes.Get(2));
    Ptr<WaypointMobilityModel> wp3 = DynamicCast<WaypointMobilityModel>(
        uavNodes.Get(2)->GetObject<MobilityModel>());
    wp3->AddWaypoint(Waypoint(Seconds(0),    Vector(50,   0,   50)));
    wp3->AddWaypoint(Waypoint(Seconds(50),   Vector(550,  0,   50)));
    wp3->AddWaypoint(Waypoint(Seconds(100),  Vector(550,  500, 50)));
    wp3->AddWaypoint(Waypoint(Seconds(150),  Vector(50,   500, 50)));
    wp3->AddWaypoint(Waypoint(Seconds(200),  Vector(50,   0,   50)));
    wp3->AddWaypoint(Waypoint(Seconds(simTime), Vector(50, 0,   50)));

    MobilityHelper mobUav4;
    mobUav4.SetMobilityModel("ns3::WaypointMobilityModel");
    mobUav4.Install(uavNodes.Get(3));
    Ptr<WaypointMobilityModel> wp4 = DynamicCast<WaypointMobilityModel>(
        uavNodes.Get(3)->GetObject<MobilityModel>());
    wp4->AddWaypoint(Waypoint(Seconds(0),    Vector(0,    50,  50)));
    wp4->AddWaypoint(Waypoint(Seconds(50),   Vector(500,  50,  50)));
    wp4->AddWaypoint(Waypoint(Seconds(100),  Vector(500,  550, 50)));
    wp4->AddWaypoint(Waypoint(Seconds(150),  Vector(0,    550, 50)));
    wp4->AddWaypoint(Waypoint(Seconds(200),  Vector(0,    50,  50)));
    wp4->AddWaypoint(Waypoint(Seconds(simTime), Vector(0,  50,  50)));

    // 地面基站 - 固定位置
    MobilityHelper mobGs;
    mobGs.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobGs.Install(gsNodes);
    gsNodes.Get(0)->GetObject<ConstantPositionMobilityModel>()->SetPosition(Vector(0,   0,   0));
    gsNodes.Get(1)->GetObject<ConstantPositionMobilityModel>()->SetPosition(Vector(500, 500, 0));

    // 干扰源
    if (scenario == 2) {
        MobilityHelper mobIntf;
        mobIntf.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mobIntf.Install(interfererNodes);
        interfererNodes.Get(0)->GetObject<ConstantPositionMobilityModel>()->SetPosition(Vector(250, 250, 10));
    }

    // ------------------------------------------------------------------------
    // WiFi 物理层配置 (PHY)
    // ------------------------------------------------------------------------
    WifiHelper wifi;
    // ns-3.48 API: WIFI_STANDARD_80211n 替代 WIFI_STANDARD_80211n_2_4GHZ
    wifi.SetStandard(WIFI_STANDARD_80211n);

    YansWifiChannelHelper channel;
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");

    double txPower = 20.0;
    std::string propModelName;
    switch (scenario) {
        case 0:
            channel.AddPropagationLoss("ns3::FriisPropagationLossModel");
            txPower = 20.0;
            propModelName = "Friis";
            NS_LOG_INFO("场景=正常LOS: Friis模型, TxPower=" << txPower << "dBm");
            break;

        case 1:
            channel.AddPropagationLoss("ns3::ThreeLogDistancePropagationLossModel",
                "Distance1", DoubleValue(100.0),
                "Exponent1", DoubleValue(2.0),
                "Distance2", DoubleValue(300.0),
                "Exponent2", DoubleValue(3.5),
                "Exponent3", DoubleValue(5.0));
            txPower = 20.0;
            propModelName = "ThreeLogDistance";
            NS_LOG_INFO("场景=建筑物遮挡NLOS: ThreeLogDistance模型(2.0/3.5/5.0)");
            break;

        case 2:
            channel.AddPropagationLoss("ns3::FriisPropagationLossModel");
            txPower = 20.0;
            propModelName = "Friis";
            NS_LOG_INFO("场景=同频干扰: Friis模型 + 干扰节点");
            break;

        case 3:
            channel.AddPropagationLoss("ns3::FriisPropagationLossModel");
            txPower = 12.0;
            propModelName = "Friis";
            NS_LOG_INFO("场景=降雨衰减: Friis模型, TxPower=" << txPower << "dBm(模拟雨衰)");
            break;
    }

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.Set("TxPowerStart",    DoubleValue(txPower));
    phy.Set("TxPowerEnd",      DoubleValue(txPower));
    phy.Set("TxPowerLevels",   UintegerValue(1));
    phy.Set("RxSensitivity",   DoubleValue(-96.0));
    phy.Set("ChannelSettings", StringValue("{0, 20, BAND_2_4GHZ, 0}"));
    // ns-3.48 中不需要 EnableMonitoring()

    // ------------------------------------------------------------------------
    // MAC 层
    // ------------------------------------------------------------------------
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    // ------------------------------------------------------------------------
    // 速率控制与网卡安装
    // ------------------------------------------------------------------------
    NetDeviceContainer devices;

    // UAV2: MinstrelHt 自适应高码率
    wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");
    devices.Add(wifi.Install(phy, mac, uavNodes.Get(1)));

    // 其他节点: 固定 HtMcs0
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
        "DataMode",    StringValue("HtMcs0"),
        "ControlMode", StringValue("HtMcs0"));
    devices.Add(wifi.Install(phy, mac, uavNodes.Get(0)));
    devices.Add(wifi.Install(phy, mac, uavNodes.Get(2)));
    devices.Add(wifi.Install(phy, mac, uavNodes.Get(3)));
    devices.Add(wifi.Install(phy, mac, gsNodes.Get(0)));
    devices.Add(wifi.Install(phy, mac, gsNodes.Get(1)));

    if (scenario == 2) {
        devices.Add(wifi.Install(phy, mac, interfererNodes.Get(0)));
    }

    // ------------------------------------------------------------------------
    // 网络层 - AODV
    // ------------------------------------------------------------------------
    AodvHelper aodv;
    aodv.Set("HelloInterval",       TimeValue(Seconds(1.0)));
    aodv.Set("AllowedHelloLoss",    UintegerValue(2));
    aodv.Set("RreqRetries",         UintegerValue(5));
    aodv.Set("ActiveRouteTimeout",  TimeValue(Seconds(10.0)));
    aodv.Set("DestinationOnly",     BooleanValue(false));
    aodv.Set("EnableHello",         BooleanValue(true));
    aodv.Set("GratuitousReply",     BooleanValue(true));

    InternetStackHelper stack;
    stack.SetRoutingHelper(aodv);
    stack.Install(allNodes);

    // ------------------------------------------------------------------------
    // IP 地址分配
    // ------------------------------------------------------------------------
    Ipv4AddressHelper address("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    NS_LOG_INFO("IP地址分配完成:");
    for (uint32_t i = 0; i < allNodes.GetN(); ++i) {
        NS_LOG_INFO("  Node " << i << " -> " << interfaces.GetAddress(i));
    }

    // ------------------------------------------------------------------------
    // 应用层流量
    // ------------------------------------------------------------------------
    // 感知数据流: UAV2 -> UAV1
    uint16_t sensePort = 5000;
    UdpServerHelper senseServer(sensePort);
    ApplicationContainer senseServerApp = senseServer.Install(uavNodes.Get(0));
    senseServerApp.Start(Seconds(1.0));
    senseServerApp.Stop(Seconds(simTime));

    UdpClientHelper senseClient(interfaces.GetAddress(0), sensePort);
    senseClient.SetAttribute("MaxPackets", UintegerValue(10000000));
    senseClient.SetAttribute("Interval",   TimeValue(MicroSeconds(100)));
    senseClient.SetAttribute("PacketSize", UintegerValue(1472));
    ApplicationContainer senseClientApp = senseClient.Install(uavNodes.Get(1));
    senseClientApp.Start(Seconds(2.0));
    senseClientApp.Stop(Seconds(simTime));

    // 控制信令流: UAV1 -> GS1
    uint16_t ctrlPort = 6000;
    UdpServerHelper ctrlServer(ctrlPort);
    ApplicationContainer ctrlServerApp = ctrlServer.Install(gsNodes.Get(0));
    ctrlServerApp.Start(Seconds(1.0));
    ctrlServerApp.Stop(Seconds(simTime));

    UdpClientHelper ctrlClient(interfaces.GetAddress(4), ctrlPort);
    ctrlClient.SetAttribute("MaxPackets", UintegerValue(100000));
    ctrlClient.SetAttribute("Interval",   TimeValue(MilliSeconds(100)));
    ctrlClient.SetAttribute("PacketSize", UintegerValue(64));
    ApplicationContainer ctrlClientApp = ctrlClient.Install(uavNodes.Get(0));
    ctrlClientApp.Start(Seconds(2.0));
    ctrlClientApp.Stop(Seconds(simTime));

    // 状态汇报流: UAV3 -> GS2
    uint16_t reportPort = 7000;
    UdpServerHelper reportServer(reportPort);
    ApplicationContainer reportServerApp = reportServer.Install(gsNodes.Get(1));
    reportServerApp.Start(Seconds(1.0));
    reportServerApp.Stop(Seconds(simTime));

    UdpClientHelper reportClient(interfaces.GetAddress(5), reportPort);
    reportClient.SetAttribute("MaxPackets", UintegerValue(100000));
    reportClient.SetAttribute("Interval",   TimeValue(Seconds(1.0)));
    reportClient.SetAttribute("PacketSize", UintegerValue(128));
    ApplicationContainer reportClientApp = reportClient.Install(uavNodes.Get(2));
    reportClientApp.Start(Seconds(5.0));
    reportClientApp.Stop(Seconds(simTime));

    // 干扰源流量
    if (scenario == 2) {
        uint16_t intfPort = 9999;
        UdpServerHelper intfServer(intfPort);
        ApplicationContainer intfServerApp = intfServer.Install(uavNodes.Get(0));
        intfServerApp.Start(Seconds(1.0));
        intfServerApp.Stop(Seconds(simTime));

        UdpClientHelper intfClient(interfaces.GetAddress(0), intfPort);
        intfClient.SetAttribute("MaxPackets", UintegerValue(10000000));
        intfClient.SetAttribute("Interval",   TimeValue(MicroSeconds(50)));
        intfClient.SetAttribute("PacketSize", UintegerValue(1472));
        ApplicationContainer intfClientApp = intfClient.Install(interfererNodes.Get(0));
        intfClientApp.Start(Seconds(3.0));
        intfClientApp.Stop(Seconds(simTime));
    }

    // ------------------------------------------------------------------------
    // Trace 回调配置
    // ------------------------------------------------------------------------
    g_rssiLog.open("rssi-log-ai3.csv", std::ios::out);
    g_rssiLog << "Time_s,RxNodeId,FreqMHz,ModeName,RSSI_dBm,Noise_dBm\n";

    // ns-3.48 MonitorSnifferRx 需要 7 个非 context 参数的回调
    Config::Connect("/NodeList/*/DeviceList/*/Phy/MonitorSnifferRx",
                    MakeCallback(&PhyRxTrace));

    g_envLog.open("env-snapshot-ai3.csv", std::ios::out);
    g_envLog << "Time_s,FromNode,ToNode,Distance_m,PathLoss_dB\n";
    Simulator::Schedule(Seconds(1.0), &LogEnvironmentSnapshot, allNodes);

    // ------------------------------------------------------------------------
    // AI3-Trace 初始化
    // 契合调研3 : DeepMIMO / RadioMapSeer / WAIR-D 公开数据集格式
    // ------------------------------------------------------------------------
    // DeepMIMO 格式: 发射机/接收机位置、信道功率、LoS/NLoS指示
    g_ai3Trace.open("ai3-trace-deepmimo.csv", std::ios::out);
    g_ai3Trace << "Scenario,Time_s,TxNode,Tx_X,Tx_Y,Tx_Z,RxNode,Rx_X,Rx_Y,Rx_Z,"
                  "Distance_m,PathLoss_dB,LoS_Status,ScenarioTag,Freq_MHz\n";

    // RadioMapSeer / WAIR-D 元数据: 场景标签、障碍物地图索引、仿真参数
    g_ai3Meta.open("ai3-meta-radiomapseer.csv", std::ios::out);
    g_ai3Meta << "# RadioMapSeer / DeepMIMO / WAIR-D Compatible Metadata\n";
    g_ai3Meta << "ScenarioID,ScenarioName,NumUav,NumGs,TotalNodes,SimTime_s,"
                 "FreqBand_GHz,CenterFreq_MHz,TxPower_dBm,ChannelWidth_MHz,"
                 "StdName,PropagationModel,AdditionalNotes\n";
    g_ai3Meta << scenario << ","
              << scenarioNames[scenario] << ","
              << nUav << "," << nGs << "," << allNodes.GetN() << ","
              << simTime << ",2.4,2400," << txPower << ",20,"
              << "802.11n," << propModelName << ",";
    switch (scenario) {
        case 0: g_ai3Meta << "DeepMIMO LoS baseline"; break;
        case 1: g_ai3Meta << "RadioMapSeer NLOS with virtual buildings (200-300,200-300,0-45)"; break;
        case 2: g_ai3Meta << "WAIR-D co-channel interference scenario"; break;
        case 3: g_ai3Meta << "DeepMIMO ITU-R P.838 rain attenuation at 2.4GHz"; break;
    }
    g_ai3Meta << "\n";
    g_ai3Meta.flush();

    Simulator::Schedule(Seconds(1.0), &Ai3TraceSnapshot, allNodes, scenario);

    // CARLA 轨迹输出 (10Hz, 契合调研3 CARLA 双目视觉+轨迹数据集)
    g_carlaOut.open("ai3-trajectory-carla.csv", std::ios::out);
    g_carlaOut << "Time_s,NodeId,Pos_X,Pos_Y,Pos_Z,Vel_X,Vel_Y,Vel_Z\n";
    Simulator::Schedule(Seconds(0.0), &LogCarlaTrajectory, allNodes);

    if (enablePcap) {
        phy.EnablePcap("low-altitude-manet-ai3", devices.Get(0));
        phy.EnablePcap("low-altitude-manet-ai3", devices.Get(1));
    }

    // ------------------------------------------------------------------------
    // FlowMonitor
    // ------------------------------------------------------------------------
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // ------------------------------------------------------------------------
    // 运行仿真
    // ------------------------------------------------------------------------
    Simulator::Stop(Seconds(simTime));
    NS_LOG_INFO("仿真开始...");
    Simulator::Run();
    NS_LOG_INFO("仿真结束.");

    // ------------------------------------------------------------------------
    // FlowMonitor 结果输出
    // ------------------------------------------------------------------------
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    std::cout << "\n========== 端到端流量统计 ==========" << std::endl;
    std::cout << std::left << std::setw(8)  << "FlowID"
              << std::setw(22) << "Source"
              << std::setw(22) << "Destination"
              << std::setw(10) << "TxPkts"
              << std::setw(10) << "RxPkts"
              << std::setw(10) << "Lost"
              << std::setw(14) << "Throughput"
              << std::setw(14) << "AvgDelay"
              << std::setw(12) << "AvgJitter"
              << std::endl;
    std::cout << std::string(124, '-') << std::endl;

    for (const auto& kv : stats) {
        FlowId flowId = kv.first;
        const FlowMonitor::FlowStats& st = kv.second;
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);

        double throughput = st.rxBytes * 8.0 / simTime / 1e6;
        double avgDelay   = (st.rxPackets > 0) ? st.delaySum.GetSeconds() / st.rxPackets : 0;
        double avgJitter  = (st.rxPackets > 1) ? st.jitterSum.GetSeconds() / (st.rxPackets - 1) : 0;

        // ns-3.48: Ipv4Address 无 ToString(), 使用流操作符
        std::ostringstream srcAddrOss, dstAddrOss;
        srcAddrOss << t.sourceAddress;
        dstAddrOss << t.destinationAddress;

        std::cout << std::left << std::setw(8)  << flowId
                  << std::setw(22) << (srcAddrOss.str() + ":" + std::to_string(t.sourcePort))
                  << std::setw(22) << (dstAddrOss.str() + ":" + std::to_string(t.destinationPort))
                  << std::setw(10) << st.txPackets
                  << std::setw(10) << st.rxPackets
                  << std::setw(10) << st.lostPackets
                  << std::setw(14) << std::fixed << std::setprecision(3) << throughput
                  << std::setw(14) << std::fixed << std::setprecision(6) << avgDelay
                  << std::setw(12) << std::fixed << std::setprecision(6) << avgJitter
                  << std::endl;
    }

    monitor->SerializeToXmlFile("flowmon-results-ai3.xml", true, true);

    // ------------------------------------------------------------------------
    // 清理
    // ------------------------------------------------------------------------
    Simulator::Destroy();
    g_rssiLog.close();
    g_envLog.close();
    g_ai3Trace.close();
    g_ai3Meta.close();
    g_carlaOut.close();

    std::cout << "\n========== 仿真输出文件 ==========" << std::endl;
    std::cout << "  rssi-log-ai3.csv         : PHY层RSSI/Noise时序数据" << std::endl;
    std::cout << "  env-snapshot-ai3.csv     : 节点距离与理论路径损耗快照" << std::endl;
    std::cout << "  flowmon-results-ai3.xml  : FlowMonitor端到端统计(XML)" << std::endl;
    std::cout << "  ai3-trace-deepmimo.csv   : AI3-Trace (契合调研3 DeepMIMO格式)" << std::endl;
    std::cout << "     列: Scenario,Time,TxNode,Tx(x,y,z),RxNode,Rx(x,y,z)," << std::endl;
    std::cout << "          Distance,PathLoss,LoS_Status,ScenarioTag,FreqMHz" << std::endl;
    std::cout << "  ai3-meta-radiomapseer.csv: AI3-Metadata (契合调研3 RadioMapSeer/WAIR-D)" << std::endl;
    std::cout << "     含: 场景参数、节点数、频段、传播模型、备注" << std::endl;
    std::cout << "  ai3-trajectory-carla.csv : CARLA轨迹数据 (10Hz, 契合调研3 CARLA双目数据集)" << std::endl;
    std::cout << "     列: Time,NodeId,Pos(x,y,z),Vel(x,y,z)" << std::endl;
    if (enablePcap) {
        std::cout << "  low-altitude-manet-ai3-*.pcap: Wireshark抓包文件" << std::endl;
    }
    std::cout << "==================================\n" << std::endl;

    return 0;
}
