# HTTP/3 over QUIC vs HTTP/2 over TCP: Where the Latency Comes From, How Much It Delivers, and Where It Reverses

HTTP/3 over QUIC buys latency in two places and loses it in a third. The first is the handshake: a cold HTTP/2 connection over TCP plus TLS 1.3 needs two round trips before the request can leave — one for TCP, one for TLS — while QUIC carries the TLS handshake inside its own transport handshake and becomes usable after one round trip; on resumption QUIC can carry application data in the first flight with 0-RTT [1][2][3]. The second is loss isolation: TCP exposes a single ordered byte stream, so a lost segment stalls every HTTP/2 stream behind it, whereas QUIC carries stream identity and byte offsets in STREAM frames, letting a receiver deliver other streams' contiguous data while the damaged stream waits [4][5].

The measured payoff is real but conditional. Google's production deployment found mean Search latency 8.0% lower on desktop and 3.6% lower on mobile, with much larger reductions at the top of the distribution — 16.7% at p99 on desktop and 14.3% on mobile [3]. Cloudflare's draft-era comparison reported average time-to-first-byte of 176 ms for HTTP/3 against 201 ms for HTTP/2, a 12.4% difference [6]. But a peer-reviewed study of production Google, Facebook and Cloudflare endpoints found no page-level benefit from QUIC's removal of cross-stream head-of-line blocking [7], and on a 1 Gbps testbed Chrome over QUIC averaged 478 Mbps, 45.2% slower than HTTP/2 [8]. The pattern is that HTTP/3 wins on cold or lossy connections carrying small-to-medium objects, reaches rough parity on clean links with large objects, and loses on high-bandwidth transfers and in per-byte CPU cost.

## The cold-start advantage is roughly one round trip, and conditions can eat it

With TLS 1.3, the TCP path costs two complete round trips before an HTTP request can be sent: one for the TCP three-way handshake and one for the TLS 1.3 handshake [2]. QUIC merges the transport and cryptographic setup, so absent packet loss a new connection is secured within a single round trip, and a returning client can often send application data immediately in 0-RTT [1]. A testbed comparison measured exactly this accounting — TCP+TLS at two round trips, QUIC at one [2] — and Google described its own pre-standard QUIC handshake as one round trip against at least one RTT for TCP plus TLS round trips [3].

The saving was larger against TLS 1.2, which needed two round trips of its own and made the total gap up to three RTTs; older QUIC studies that used TLS 1.2 therefore report a bigger setup advantage than applies to a modern HTTP/2 stack with TLS 1.3 [9][3]. TCP Fast Open can remove the TCP round trip, but it was not widely deployed in the measurement studies, and TLS 1.3 early data over TCP can carry application data before the handshake completes but still waits for the TCP handshake first [2][10].

| Phase | HTTP/2 over TCP + TLS 1.3 | HTTP/3 over QUIC |
|---|---|---|
| Cold connection before first request | 2 RTT: TCP handshake plus TLS 1.3 handshake [2] | 1 RTT: transport and TLS combined [1][2] |
| Resumed connection | 1 RTT; TLS early data can follow but TCP handshake remains [10] | 0 RTT with a valid ticket [1][10] |
| Extra-RTT triggers | TCP retransmission, TLS HelloRetryRequest | Retry, version negotiation, HelloRetryRequest, oversized certificate [11][12] |

Google measured handshake latency at the server from the first SYN or client hello to completion and found the TLS/TCP mean rising roughly linearly with minimum RTT while the QUIC mean stayed nearly flat [3]. That is the cleanest statement of the mechanism: the saved round trip is worth more the higher the path RTT.

Two constraints reduce the one-RTT claim in practice. Before validating the client address, a QUIC server may send at most three times the bytes it received, so a certificate chain larger than that budget forces additional round trips [11][12]. A 2022 scan of 272,000 QUIC services found that at a 1362-byte client Initial, 38% of handshakes required multiple round trips, 35% of certificate chains exceeded the three-times limit, and certificate compression — median reduction about 65% — brought 99% of chains under it [12]. Browsers at the time sent 1250 to 1357-byte Initials [12]. Separately, Retry, version negotiation and a TLS HelloRetryRequest each add a round trip [11]. The one-RTT handshake is a property of a well-configured server with a small enough certificate, not an unconditional property of QUIC.

## Multiplexing removes cross-stream blocking, but the page-level gain is conditional

The mechanism is unambiguous in the specifications. HTTP/2 multiplexes independent streams over one TCP connection, but TCP sees only an opaque ordered byte stream; if packet 2 is lost while packets 1 and 3 arrive, TCP holds packet 3 in its receive buffer even when it contains stream-1 data that follows contiguously, and retransmission takes at least one round trip [5]. QUIC moves stream identity and byte offsets into the transport: STREAM frames carry a stream ID and offset, the receiver can deliver contiguous data for unaffected streams, and the specification states that QUIC "does not provide any means of ensuring ordering between bytes on different streams" [4][5].

That removes transport-layer head-of-line blocking across streams, not within them. QUIC still requires each stream to be delivered as an ordered byte stream and still buffers out-of-order data up to the flow-control limit, so a gap inside one stream blocks that stream exactly as TCP would [4]. A packet can also carry frames from several streams — QUIC permits coalescing multiple packets, even of different types, into one UDP datagram — so one loss can re-couple the fates of the streams it carried; a 2020 practitioner analysis notes that implementations therefore rarely put data from multiple streams into a single packet [4][5].

QUIC's loss recovery differs as well, and that difference is part of the tail-latency story. Packet numbers encode transmission order while STREAM offsets encode delivery order, so a loss epoch closes as soon as any later-sent packet is acknowledged, whereas TCP must wait for the sequence-number gap to fill and can stall across several round trips if the same segment is repeatedly lost [13]. QUIC ACK frames carry many ranges rather than TCP's three SACK ranges, and the recommended loss declaration is three packets or 9/8 of the RTT [13]. These are specification mechanics, not measurements, but they mean the latency after a loss depends on both what the loss blocks and how quickly it is detected.

What the experiments show is more modest than the mechanism suggests:

- A peer-reviewed study of production Google, Facebook and Cloudflare endpoints found that QUIC's removal of head-of-line blocking "has little impact on web-page performance in practice": HTTP/3 showed no Speed Index advantage over HTTP/2 under added loss for Google or Facebook, and Cloudflare's apparent gain came from a FIFO stream scheduler rather than from HOL removal [7].
- A draft-era Lighthouse benchmark found HTTP/3 behind HTTP/2 at baseline and only overtaking as loss increased at a fixed 300 ms delay, with results unstable beyond 1.4% loss [9].
- A 2025 preprint with 20 concurrent 1 MB streams on a fixed 10 Mbps link found the worst-stream total request time at 12% loss was 113.42 s for HTTP/2 against 20.98 s for HTTP/3, an 81.5% reduction, but HTTP/3 only clearly pulled ahead beyond about 8% loss [14].
- An MQTT-over-QUIC experiment using pre-standard QUIC Crypto measured aggregate delivery latency at 10% loss of 1.3833 ms versus 3.114 ms for TCP on a wired link (55.6% lower), 7.11 versus 11.22 ms on wireless, and 33.0 versus 44.4 ms long-distance; the advantage shrank to roughly 13–16% at 50% loss [15].
- A reproducible testbed that tuned TCP to match QUIC's initial window and pacing found that after subtracting the one-RTT handshake advantage, the remaining differences fell below a single RTT on lossless DSL and LTE links; the authors explicitly say they *assume*, rather than measure, that QUIC's edge on a 6% loss link comes partly from avoided head-of-line blocking [2].

No experiment in this set measures per-stream completion latency after a single isolated packet loss across N concurrent streams for HTTP/2 and HTTP/3. The closest measurement is the worst-stream aggregate at 8–12% loss [14]; the adversarial case where one QUIC packet carries frames from several streams is described mechanically but not measured [5][4].

## Measured outcomes: TTFB and page load improve mainly at high RTT and under loss

The headline deployment and testbed numbers span a wide range because they measure different things under different conditions.

| Setting | What was measured | Result |
|---|---|---|
| Google production, 2016–17, QUIC v35 [3] | Mean Search latency | 8.0% lower desktop, 3.6% lower mobile |
| Google production [3] | Search p99 | 16.7% lower desktop, 14.3% lower mobile |
| Cloudflare, HTTP/3 drafts 23–27 [6] | Average TTFB | 176 ms vs 201 ms |
| Cloudflare [6] | 1 MB page load | 2.33 s vs 2.30 s, HTTP/3 slower |
| Yu and Benson, WWW '21, h3-29 [7] | 100 KB object at 10 Mbps | HTTP/3 faster; parity at 1 MB and above |
| Saif et al., draft 27 [9] | Lighthouse aggregate, baseline | 65 vs 87, HTTP/3 worse |
| Liu et al., 2025 preprint [14] | 20 × 1 MB streams, 12% loss | 20.98 s vs 113.42 s worst stream |
| 1 Gbps testbed, Chrome [8] | 1 GB download | 18.60 s vs 9.32 s, HTTP/3 slower |

The gains scale with RTT. Google's country-level split put desktop Search latency reduction at 13.2% in India against 1.3% in South Korea, consistent with a fixed round-trip saving mattering more on longer paths [3]. The Yu and Benson production study found HTTP/3 consistently faster for 100 KB single objects at 10 Mbps, essentially equivalent for larger objects, and attributed the small-object gain to the one-RTT handshake advantage, which was significant below 1 MB at 10 Mbps and negligible above it [7]. Cloudflare saw a 15 KB test page load faster over HTTP/3 (443 ms versus 458 ms) but a 1 MB page slightly slower (2.33 s versus 2.30 s) [6].

The gains also scale with loss, but only past a threshold. The 2025 preprint's crossover was around 8% loss for 20 concurrent streams [14]; the Lighthouse benchmark saw HTTP/3 overtake HTTP/2 as loss rose at a fixed 300 ms delay, while its own results became unstable beyond 1.4% loss [9]. At the other end, the MQTT experiment showed the QUIC advantage shrinking as loss rose from 10% to 50% [15].

Every large HTTP/3-versus-HTTP/2 figure above comes from a pre-standard or draft implementation: Google's QUIC v35 data are from 2016–2017 [3], Cloudflare tested drafts 23–27 [6], the production study used h3-29 [7], and the Lighthouse benchmark used draft 27 [9]. A 2026 journal testbed report puts connection setup at 132 ms for TCP/HTTP/2 against 54 ms for QUIC/HTTP/3 and page load at 1,248 ms against 912 ms, but it does not state the RTT or loss profile behind those figures, attributes a cold-connection improvement partly to 0-RTT, and cites a reference list largely unrelated to QUIC transport; it is a weak data point rather than corroboration [16].

## Tail latency: the gains concentrate at p95/p99, and the field data are pre-standard

The only HTTP/3-versus-HTTP/2 percentile table in this set is Google's 2016–2017 Search measurement. Positive values mean lower latency for QUIC:

| Percentile | Desktop Search | Mobile Search |
|---|---|---|
| p1 | 0.4% | −0.6% |
| p5 | 1.3% | −0.3% |
| p10 | 1.4% | 0.3% |
| p50 | 1.5% | 0.5% |
| p90 | 5.8% | 4.5% |
| p95 | 10.3% | 8.8% |
| p99 | 16.7% | 14.3% |

The shape matters more than any single number. The median moves by about 1.5% on desktop and half a percent on mobile, while p99 moves by 16.7% and 14.3% [3]. That is what a mechanism based on connection setup and loss recovery should look like: it does little for a typical request that reuses a warm connection and a lot for the slowest requests where a handshake, a retransmission or a stalled stream dominates. On mobile, the lowest percentiles were slightly *worse* for QUIC, which Google attributed in part to user-space scheduler and process costs [3].

Those numbers are pre-IETF QUIC v35, and no 2023–2026 CDN, browser-vendor or operator A/B dataset reporting HTTP/3 versus HTTP/2 p95 or p99 was found. Cloudflare reported only averages, with no percentiles and no RTT, loss or bandwidth stratification [6]. A 2026 Frontiers in Computer Science study provides a properly reported percentile table, but for TLS 1.3 0-RTT over TCP only: at 50 ms emulated RTT, classical 0-RTT had p99 of 54.1 ms against 142.5 ms for a post-quantum full handshake, a lab result with no QUIC arm [17]. A 2026 single-host lab comparison of raw QUIC against TCP measured p95 of 0.36–0.39 ms for QUIC against 0.20–0.21 ms for TCP — roughly 1.8 times worse, calculated from those figures — but it tested neither HTTP/3 nor HTTP/2 and ran both endpoints on one WSL machine [18]. The CESNET-QUICEXT-25 dataset could support deriving handshake round-trip and 0-RTT distributions from an ISP vantage point, but it has no HTTP/2 comparison and no TTFB field [19].

The practical consequence is that the tail benefit is well supported in direction and weakly supported in current magnitude. The standard error on "HTTP/3 lowers p99 by about 15%" is large for a 2026 deployment.

## Where HTTP/3 is slower: high bandwidth, CPU cost, single-stream loss, UDP filtering

### High-bandwidth links above roughly 500 Mbps

On a controlled 1 Gbps Ethernet testbed with CUBIC on both stacks and oversized socket buffers, Chrome over QUIC averaged barely 478 Mbps and was 45.2% slower than HTTP/2; a 1 GB download took 18.60 s against 9.32 s, and on a Pixel 5 it took 78.65 s against 37.11 s [8]. The regression appeared above roughly 500 Mbps for Chrome and 600 Mbps for lightweight clients; below those rates the two stacks were similar. Page load time averaged 3.0% worse over HTTP/3, with a tail up to 74.9% worse [8]. The authors attribute this to receiver-side CPU and packet-processing overhead — 744,000 packets versus 58,000, packet RTT of 16.2 ms versus 1.9 ms, and user-space ACK processing — rather than to the wire format [8].

On a 10 Gbit/s testbed, QUIC goodput ranged from 52 to 3004 Mbit/s across client–server library pairings while TCP/TLS reached 8010 Mbit/s in the same setup [20]. The default 208 KiB UDP receive buffer caused packet loss; scaling it about 16 times lifted LSQUIC by 8.7% to 3250 Mbit/s and quiche by 3% to 2530 Mbit/s [20]. A 2025 study of five implementations found HTTP/3 goodput between 1.22 and 4.17 Gbit/s for the same hardware depending on pairing — a spread of more than three times — and only jumbo frames let one pairing reach 9.97 Gbit/s [21].

### Per-byte CPU cost

Google reported server CPU at about 3.5 times its TLS/TCP baseline initially and about 2 times after optimization [3]. In the 1 Gbps browser test, Chrome used 96.9% CPU on QUIC against 77.5% on HTTP/2, and Chrome-QUIC approached 100% CPU at only 200 Mbps [8]. Removing AES hardware acceleration cost 11% of goodput for LSQUIC and 51% for quiche, and packet I/O rather than cryptography was the dominant cost [20]. This is a cost per byte, so it becomes a latency penalty under load, on weak client hardware, or when a single connection is trying to fill a fast link.

### Intra-stream blocking and packet coalescing

HTTP/3 cannot beat TCP's ordering when only one stream matters. A single large download is one stream; a gap in it blocks that stream exactly as TCP would, and if a packet carries frames from several streams, the loss blocks all of them [4][5]. That is why the measured loss benefit appears with many concurrent streams and fades for a single object [7][14], and why implementations tend to avoid mixing streams in one packet [5].

### Parity or reversal at low loss and small object sizes

At baseline with no added impairment, the draft-era Lighthouse benchmark scored HTTP/3 at 65 against 87 for HTTP/2, with Largest Contentful Paint "decisively worse" for HTTP/3, and HTTP/3 sent almost twice as many packets [9]. Cloudflare's own blog loaded 1–4% slower over HTTP/3 across regions, a result the post attributes partly to a congestion-control mismatch — HTTP/2 ran BBR v1 while HTTP/3 ran CUBIC [6]. After the one-RTT handshake advantage was subtracted on lossless links, the tuned testbed found differences below a single RTT [2].

### UDP filtering and middleboxes

QUIC runs over UDP, and some networks interfere with it. An Internet Society Pulse summary of a USENIX Security 2025 measurement study reports deep-packet inspection and filtering of UDP carrying QUIC payloads in select autonomous systems in Iran [22]. That source establishes that filtering exists; it gives no HTTP/3-versus-HTTP/2 latency, failure-rate, timeout or fallback-time measurements. The architectural cost would be a delayed first connection while a client times out on QUIC and retries over TCP, but no measurement found here puts a number on that delay.

| Condition | Direction of measured effect |
|---|---|
| Cold connection, high RTT | HTTP/3 lower latency; about one RTT saved [2][3] |
| Loss at or above roughly 8% with many streams | HTTP/3 far ahead; 81.5% at 12% loss in one testbed [14] |
| Clean link, object at or above 1 MB | Roughly equal, or HTTP/2 slightly ahead [7][6] |
| Link at or above 500 Mbps–1 Gbps | HTTP/3 45.2% slower at 1 Gbps [8] |
| 10 Gbit/s with varied implementations | 1.22–4.17 Gbit/s against 8.01 Gbit/s TCP/TLS [20][21] |
| UDP-filtered network | Failure or fallback; latency cost unquantified [22] |

## 0-RTT saves another round trip, at a replay cost

QUIC 0-RTT is available only after a prior connection, uses a TLS session ticket, and is capped by TLS 1.3 at seven days from the original connection; the application protocol must define a replay-safe 0-RTT profile, and the server can reject early data, in which case the client must reset the affected streams and retry after the handshake [1]. Cloudflare's operational account adds the practical constraints: 0-RTT is not enabled by default, early data lacks forward secrecy, an on-path attacker can capture and replay it, and the company therefore rejects obviously non-idempotent POST and PUT requests sent in 0-RTT, adds `Early-Data: 1` when forwarding, and lets origins answer `425 Too Early` to force a retry [10]. A cryptographic treatment reaches the same conclusion from the primitive level: 0-RTT has no inherent replay protection without application-layer countermeasures and provides only weak forward secrecy, while a full first handshake costs one RTT [23].

On the benefit side, Google reported that 88% of desktop Search connections over QUIC used 0-RTT, 85% for desktop video, 68% for the mobile app and 65% for mobile YouTube, and counted a 0-RTT handshake as 0 ms in its handshake-latency measurement [3]. That establishes uptake, not the marginal saving. No controlled same-testbed comparison of QUIC 0-RTT against TCP with TLS 1.3 early data was found; vendor commentary puts the saving at 50–200 ms for a returning client on a high-latency link without a cited benchmark [24]. The only measured tail data on 0-RTT in this set is TCP-only and concerns post-quantum key exchange, where a full post-quantum handshake raised p99 from 54.1 ms to 142.5 ms at 50 ms RTT [17]. What is missing is the rejection rate: how often servers actually refuse early data in deployment, and what that costs, is not quantified in the sources found here.

## Attribution: implementation and congestion control explain much of the spread

Several results that look like protocol verdicts are really stack verdicts.

The clearest is the tuned-TCP experiment: giving TCP an initial window of 32, pacing, CUBIC and tuned buffers, then subtracting QUIC's one-RTT handshake advantage, reduced the difference to below one RTT on lossless links [2]. A naive comparison against stock TCP overstates QUIC's advantage by roughly the amount of TCP tuning it omits.

Congestion control can outweigh the protocol difference outright. Cloudflare's HTTP/3 deficit on its own blog coincided with HTTP/2 running BBR v1 and HTTP/3 running CUBIC, which the post itself calls not apples-to-apples [6]. In the 2025 proxy study, adding an Envoy proxy with BBR brought a severe HTTP/2 case from about 106 s down to about 10 s — better than HTTP/3 without the proxy — showing that middlebox and congestion-control configuration can dominate the transport choice [14].

Implementation quality produces spreads larger than most reported protocol effects. QUIC goodput varied from 52 to 3004 Mbit/s across library pairings on identical 10 Gbit/s hardware [20], and from 1.22 to 4.17 Gbit/s across five implementations in a separate study, where switching from HTTP/3 to HTTP/0.9 over the same QUIC transport changed throughput by up to 27% [21]. Internet-wide, four libraries — LSQUIC, quiche, gQUICHE and s2n-quic — served more than 95% of successful QUIC handshakes in a five-year scan covering 2021 to 2026, and those libraries differ in defaults and in the congestion control they ship [25]. A measured "HTTP/3 average" is therefore effectively a library-weighted average.

Client implementation matters too: in the 1 Gbps test every browser tested was slower over QUIC and used more CPU, with the gap attributed to user-space packet processing [8]. Google's own early deployment reported server CPU at 3.5 times TLS/TCP before optimization and about 2 times after, which is an implementation-maturity curve rather than a protocol constant [3].

## What to expect, and what to measure

Expect the handshake gain on cold, high-RTT connections: roughly one round trip, more on paths where TLS 1.2 is still in use, and less when the certificate chain exceeds the anti-amplification budget — certificate compression is the lever that keeps 99% of chains under it [12][2][3]. Expect the loss-isolation gain mainly when several streams compete and loss is material; the measured crossover in one testbed was around 8%, and at low loss with large objects the two stacks are close to indistinguishable [7][14].

Do not assume HTTP/3 is faster for bulk transfer on a fast last-mile link. At 1 Gbps Chrome over QUIC was 45.2% slower, and at 10 Gbit/s implementation pairing and socket-buffer tuning moved goodput by more than three times [8][20][21]. Check receiver CPU and UDP receive-buffer size before drawing a conclusion about the protocol.

For any A/B test, match congestion control, buffer tuning and implementation generation, and report cold and resumed connections separately. Averages hide the effect: Google's median Search gain was 1.5% while its p99 gain was 16.7% [3], and Cloudflare's average TTFB gain coexisted with a 1–4% regression on its own site [6]. The open measurement gaps are a post-standardization p95/p99 field comparison for HTTP/3 against HTTP/2, a single-isolated-loss per-stream experiment including packet coalescing, a quantified latency cost for UDP filtering and fallback, and a head-to-head of QUIC 0-RTT against TLS 1.3 early data over TCP.

## References

- **[1] rfc-editor.org** - "Using TLS to Secure QUIC". [Source](https://www.rfc-editor.org/rfc/rfc9001.html) (accessed 2026-11-27)
- **[2] arxiv.org** - "A Performance Perspective on Web Optimized Protocol Stacks: TCP+TLS+HTTP/2 vs. QUIC". [Source](https://arxiv.org/abs/1906.07415v1) (accessed 2026-11-27)
- **[3] dl.acm.org** - "The QUIC Transport Protocol:Design and Internet-Scale Deployment". [Source](http://dl.acm.org/ft_gateway.cfm?id=3098842&type=pdf) (accessed 2026-11-27)
- **[4] IETF Datatracker** - "RFC 9000: QUIC: A UDP-Based Multiplexed and Secure Transport". [Source](https://datatracker.ietf.org/doc/html/rfc9000) (accessed 2026-11-27)
- **[5] Web Performance Calendar** - "Head-of-Line Blocking in QUIC and HTTP/3: The Details". [Source](https://calendar.perfplanet.com/2020/head-of-line-blocking-in-quic-and-http-3-the-details/) (accessed 2026-11-27)
- **[6] Cloudflare Blog** - "Comparing HTTP/3 vs. HTTP/2 Performance". [Source](https://blog.cloudflare.com/http-3-vs-http-2/) (accessed 2026-11-27)
- **[7] cs.brown.edu** - "Dissecting Performance of Production QUIC". [Source](https://cs.brown.edu/people/tab/papers/QUIC_WWW21.pdf) (accessed 2026-11-27)
- **[8] arxiv.org** - "QUIC is not Quick Enough over Fast Internet". [Source](https://arxiv.org/html/2310.09423v2) (accessed 2026-11-27)
- **[9] arxiv.org** - "An Early Benchmark of Quality of Experience Between HTTP/2 and HTTP/3 using Lighthouse". [Source](https://arxiv.org/abs/2004.01978v3) (accessed 2026-11-27)
- **[10] Cloudflare Blog** - "Even faster connection establishment with QUIC 0-RTT resumption". [Source](https://blog.cloudflare.com/even-faster-connection-establishment-with-quic-0-rtt-resumption/) (accessed 2026-11-27)
- **[11] ietf.org** - "QUIC Deep Dive: Layering and Handshake". [Source](https://www.ietf.org/proceedings/115/slides/slides-115-tdd-sessb-layering-and-security-00.pdf) (accessed 2026-11-27)
- **[12] arxiv.org** - "On the Interplay between TLS Certificates and QUIC Performance". [Source](https://arxiv.org/pdf/2211.02421) (accessed 2026-11-27)
- **[13] rfc-editor.org** - "QUIC Loss Detection and Congestion Control". [Source](https://www.rfc-editor.org/rfc/rfc9002.html) (accessed 2026-11-27)
- **[14] arxiv.org** - "Performance Comparison of HTTP/3 and HTTP/2 with Proxy Integration". [Source](https://arxiv.org/pdf/2409.16267) (accessed 2026-11-27)
- **[15] arxiv.org** - "Implementation and Analysis of QUIC for MQTT". [Source](https://arxiv.org/pdf/1810.07730) (accessed 2026-11-27)
- **[16] jidmis.org** - "Paper Title (use style: paper title)". [Source](https://jidmis.org/index.php/jidmis/article/download/1540/601) (accessed 2026-11-27)
- **[17] frontiersin.org** - "A Hybrid Quantum-Resistant Resumption framework for zero-RTT TLS 1.3". [Source](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1848687/pdf) (accessed 2026-11-27)
- **[18] dergipark.org.tr** - "Performance Evaluation of the Next-Generation Internet Protocol QUIC". [Source](https://dergipark.org.tr/tr/download/article-file/5754789) (accessed 2026-11-27)
- **[19] Zenodo** - "CESNET-QUICEXT-25: A year-long QUIC traffic dataset with EXTended…". [Source](https://zenodo.org/records/17249078) (accessed 2026-11-27)
- **[20] arxiv.org** - "QUIC on the Highway: Evaluating Performance on High-rate Links". [Source](https://arxiv.org/pdf/2309.16395) (accessed 2026-11-27)
- **[21] doc.tm.kit.edu** - "Examining the Heterogeneous Throughput Performance Landscape of QUIC Implementations". [Source](https://doc.tm.kit.edu/2025-Examining-the-Heterogeneous-Throughput-Performance-Landscape-of-QUIC-Implementations-Koenig-et-al.pdf) (accessed 2026-11-27)
- **[22] Internet Society Pulse** - "Now There's a Way to Measure QUIC Targeting By Middleboxes | ISOC Pulse". [Source](https://pulse.internetsociety.org/en/blog/2025/12/now-theres-a-way-to-measure-quic-targeting-by-middleboxes/) (accessed 2026-11-27)
- **[23] dl.acm.org** - "Optimizing 0-RTT Key Exchange with Full Forward Security". [Source](https://dl.acm.org/doi/pdf/10.1145/3605763.3625246) (accessed 2026-11-27)
- **[24] Stack Harbor** - "TLS 1.3 0-RTT — when the latency win is worth the replay risk". [Source](https://stackharbor.com/en/knowledge-base/dnstls-tls13-zero-rtt-tradeoffs/) (accessed 2026-11-27)
- **[25] zirngibl.github.io** - "A QUIC Look Back: Longitudinal Internet-Scale Deployment and Transport Parameter Evolution". [Source](https://zirngibl.github.io/files/kempf2026quic.pdf) (accessed 2026-11-27)