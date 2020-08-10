# SubsetSampler

This is a subset sampler I built to sample subsets of network features from In-Q-Tel's NetworkML dataset.

NetworkML is an open source network role classifier that parses .pcap files and trains supervised learning models to indentify
network traffic consistent with certain devices. The orginal dataset contained 221 features per device sample. My job at In-Q-Tel
was to interpret the NetworkML's models, but this was challenging to do with such a high dimensional dataset.
Using another custom algorithm I built, I was able to group features into bins based off multicollinearity, knowing that the models only required
a subset of features from each bin to remain accurate.

An example of subset sampling from bins of features:

Given feature bins:

BIN_1: [max_frame_len, min_frame_len, 25q_frame_len]

BIN_2: [ipv4_multicast, upd_port_67_in, upd_port_67_out]

valid subset samples of these bins might be:

[max_frame_len] U [udp_port_67_in, udp_port_67_out] -> [max_frame_len, udp_port_67_in, udp_port_67_out]

or

[min_frame_len, 25q_frame_len] U [ipv4_multicast, upd_port_67_out] -> [min_frame_len, 25q_frame_len, ipv4_multicast, upd_port_67_out]

I built an earlier version of this subset sampler to brute force a sampling of a few features from each bin and fit a model to see if it was still accurate.
I ran into many issues with memory efficency because my first implementaion relied on itertools.product(). The itertool.product() docs claim to be memory efficient, but reading the
source code reveals that the lower order products are cached to generate the higher order products later. As a result I struggled to use this method with the large bins of features
because itertools.product() would eat up my memory. On the job I managed to make the sampler more memory efficeint so I could sample from larger bins of features.
On my own time, I rebuilt this super memory efficient sampler that completly ommits the use of itertools.product() and greatly reduces the amount of caching needed.
More information on itertools can be found at https://docs.python.org/3/library/itertools.html.
