import pooch
from cdlib.readwrite import *
import networkx as nx

try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None

__all__ = [
    "available_networks",
    "available_ground_truths",
    "fetch_network_data",
    "fetch_ground_truth_data",
    "fetch_network_ground_truth",
]

__networks = pooch.create(
    path=pooch.os_cache("cdlib"),
    base_url="https://github.com/GiulioRossetti/cdlib_datasets/raw/main/networks/",
    version_dev="master",
    env="cdlib_data_dir",
    registry={
        "karate_club.csv.gz": "0bac1a017e59f505f073aef6dc1783c03826a3fd4f38ffc6a50fde582948e2f2",
        "dblp.csv.gz": "6b64a23e60083fa52d64320a2cd4366ff1b33d8a89ac4fa1b30f159e88c1730c",
        "amazon.csv.gz": "aea015386f62b21ba31e097553a298fb3df610ec8dc722addb06a1a59e646cd3",
        "youtube.csv.gz": "dad87a04c648b31565448e7bef62ace569157aa58f7d2a0bb435eb490d0bf33e",
        "LFR_N100000_ad5_mc50_mu0.1.csv.gz": "54eb36e619fa7f6ff2dcbb4e5a43ee9885c8f4d9634fd9e48cdef3de963581ae",
        "LFR_N100000_ad5_mc50_mu0.2.csv.gz": "486e2cf7ce009b32ad21b4d290d6b1c2fc093a614deed101acea787e759d1b32",
        "LFR_N100000_ad5_mc50_mu0.3.csv.gz": "41f875cd0db105ee90b4f4f4e2d256dd7230f30916f498faf9e36ef724ec71f3",
        "LFR_N100000_ad5_mc50_mu0.4.csv.gz": "7eb1023a02574459b3649aefe8b3661476104e00116699b15575eca31bbd6ba2",
        "LFR_N100000_ad5_mc50_mu0.5.csv.gz": "e398d2ef53d3fb3f7f54aa3756deb1333610ce37fe5ada7ac3689d4bd0a77bc6",
        "LFR_N100000_ad5_mc50_mu0.6.csv.gz": "7ab7fd9405260c0561806cb1f23c2eb1e420b1219baa69bf9ffa7183b36fd4d3",
        "LFR_N100000_ad5_mc50_mu0.7.csv.gz": "d4a9e94f16823706a9193f5f7053fca33a8bc9ab2e3e5f68c094deb02bbb86c3",
        "LFR_N100000_ad5_mc50_mu0.8.csv.gz": "875147fd1d9ae14770b3c0e461223e5675712d06b57f591c77113dc443177f7d",
        "LFR_N100000_ad5_mc50_mu0.9.csv.gz": "ec6b9fbad572bc806239cef31f77c50afd0b10d2acdf5a047981f4756409f5b7",
        "LFR_N10000_ad5_mc50_mu0.1.csv.gz": "7ea99a90647402b55e8f87bb7dcad60875b5ec17c84c61b0b73ce6a3988137bf",
        "LFR_N10000_ad5_mc50_mu0.2.csv.gz": "d477a10a8be4cb7c8dd625bd2784af14f9b7aa0de63fc116c49c1effea017ba2",
        "LFR_N10000_ad5_mc50_mu0.3.csv.gz": "8572198101891bf69365fdcf42c59beec6798371b5f8ae47d6cc31aa4f3178ab",
        "LFR_N10000_ad5_mc50_mu0.4.csv.gz": "288f45fd4626a58d27cebaaca903ab5ca3d8eaae9e393df276097f3da4244e8a",
        "LFR_N10000_ad5_mc50_mu0.5.csv.gz": "83ba2ce5565542cc0726f254aeca28ca60ede1a237b1dd0e8c7a2e992f5b4e16",
        "LFR_N10000_ad5_mc50_mu0.6.csv.gz": "1dda29356ef25f01088ba03167890c2af33a6c13b519d778d32021154ae63eda",
        "LFR_N10000_ad5_mc50_mu0.7.csv.gz": "ba7b82d01e0c96e1b83637a44104d51d4fcee2f3ff6e0c6c12e9cde0f82538a6",
        "LFR_N10000_ad5_mc50_mu0.8.csv.gz": "3271d6c7591f92ccf8f7ec85c776e295454f282d556de63efe3f66657c4c9ec0",
        "LFR_N10000_ad5_mc50_mu0.9.csv.gz": "0f4f127af47d7a22dd5b3a641014b90eab30f1856d397b56aa5fda766ca7e217",
        "LFR_N1000_ad5_mc50_mu0.1.csv.gz": "a575245a1a3464da9dbebe85e3b68a391a43eaf9c14ce6777aa621483d5f58b3",
        "LFR_N1000_ad5_mc50_mu0.2.csv.gz": "efaafd318f1835f0e58d24309d5caa472c5dca9b72282c2b0466c0c264a6f0ea",
        "LFR_N1000_ad5_mc50_mu0.3.csv.gz": "d418d7a6dc8930b5ec7f0b33ceace4265f3077bd10272762ae1e31e506e605a8",
        "LFR_N1000_ad5_mc50_mu0.4.csv.gz": "9ee8112441a1ee9f1733e627e9b88f5048c237c5ae479203bae00736c357175f",
        "LFR_N1000_ad5_mc50_mu0.5.csv.gz": "09bd8a15ec08877dcc93d9c7fc6b80b9b5a04d7126a4f5eef25f9570fe8d6d75",
        "LFR_N1000_ad5_mc50_mu0.6.csv.gz": "90d29e105c63a01743e85def469aa301d786cebd41134e2f797a612f00dd6cca",
        "LFR_N1000_ad5_mc50_mu0.7.csv.gz": "956707249e7b1539c666c602dbba714b855884bceac1d47246c86b0666ac8afd",
        "LFR_N1000_ad5_mc50_mu0.8.csv.gz": "f798f48acf14e7a337987cad14e774cbb6984d899e3a910da94977f0865fe4ee",
        "LFR_N1000_ad5_mc50_mu0.9.csv.gz": "29a4001c29f9e35f1b8d52fa85410b676c655ffe10e622c70204b4325d182e83",
        "LFR_N50000_ad5_mc50_mu0.1.csv.gz": "bebc3640db551389b6b7a819c6512fe00cee965fb59bfcd28e7246f2707ec822",
        "LFR_N50000_ad5_mc50_mu0.2.csv.gz": "b6cafbe62d0cd64f04286d9fa29f9d8b4eb6cc788383000effa78b46ca2269db",
        "LFR_N50000_ad5_mc50_mu0.3.csv.gz": "ada823356120f5039ee3620aefb1e9a334f8aaa6711a3150823aac3cb08e18c5",
        "LFR_N50000_ad5_mc50_mu0.4.csv.gz": "e8062aeefc37e506d39a2622c27206179dcc71340b82b27ccd81f190ab9a719c",
        "LFR_N50000_ad5_mc50_mu0.5.csv.gz": "fb51f02e1a47adea3bcd6d37e6dc9c330be1f6538becb1f5ff5f8c391ce4b403",
        "LFR_N50000_ad5_mc50_mu0.6.csv.gz": "fbbcaec9fa7a762c0d62c59ca0543db3710b6ca1d8244503581b23e012c68c0e",
        "LFR_N50000_ad5_mc50_mu0.7.csv.gz": "1f2c50098d55227cbac5f007a8814ec08d451c2eaa2e5b75ebe97f61617a8d70",
        "LFR_N50000_ad5_mc50_mu0.8.csv.gz": "20e7b2d99633ddeac315576bdc3cb7dd94ac042d6cd622ef171c2cfe14a73f55",
        "LFR_N50000_ad5_mc50_mu0.9.csv.gz": "e97e9686e4b5e2efdc314bf2ed2f9bc55bd29adfb32db778e301e302511324f1",
        "LFR_N5000_ad5_mc50_mu0.1.csv.gz": "25c551db0407721897d7f3d10c087214c3593a48b5b8ce37f73fcbb61230721a",
        "LFR_N5000_ad5_mc50_mu0.2.csv.gz": "f89486f9e7df512457bacc91595317eb10f0144cf0a58d902f915fedcd43fc35",
        "LFR_N5000_ad5_mc50_mu0.3.csv.gz": "4df6ef918a258204e05a18efb3f6ff1876a11a66ed8c319d96e65b540b3f3f3c",
        "LFR_N5000_ad5_mc50_mu0.4.csv.gz": "4037d7a18792d8584e1f05fca6a1f85628e12dc428d07baaebfe1d129f4084e8",
        "LFR_N5000_ad5_mc50_mu0.5.csv.gz": "dd6efeb9ac17e845ed6ffe0488508fb847fbaf2c0b5db218a028342ea20dbf55",
        "LFR_N5000_ad5_mc50_mu0.6.csv.gz": "b1fd8e3fa9761a0a9b61250c8b5cc4add546b9e7447c71076b7ed12e6ae837f7",
        "LFR_N5000_ad5_mc50_mu0.7.csv.gz": "0c8fd15bf9121ed274a5da3c607b13f8637a8028ffcac42a6b045f301fc6320f",
        "LFR_N5000_ad5_mc50_mu0.8.csv.gz": "b171f72d53950720da049d37e1f6dae3ce2f433344e0e84967826d144de0fccb",
        "LFR_N5000_ad5_mc50_mu0.9.csv.gz": "5716355473ffb7eb9f37ea9959b05dfed6c3b910db51d71c6597bdef88584729",
    },
)

__ground_truths = pooch.create(
    path=pooch.os_cache("cdlib"),
    base_url="https://github.com/GiulioRossetti/cdlib_datasets/raw/main/ground_truth/",
    version_dev="master",
    env="cdlib_data_dir",
    registry={
        "karate_club.json.gz": "198fd42c3df9ab49e3eea5932f0d6e4cceac25db147c5108e0f8e9a4c55e11b7",
        "dblp.json.gz": "ca7dba98bd3bdc76999fd2991d1667b7b531e8bac777986247a8dcac302c085d",
        "amazon.json.gz": "c6a03909f2b14082523108be13616e5b24cfe830b96e416d784e90ab23d12bd7",
        "youtube.json.gz": "affa73428b300d896944adbbf334b77370b6fc3a43f5adbf54350fd5fafaeed5",
        "LFR_N100000_ad5_mc50_mu0.1.json.gz": "96f2022ff76448448547bbb46d0f803eee345eb96141d962de086a66b650fb81",
        "LFR_N100000_ad5_mc50_mu0.2.json.gz": "e7e4083d0f849af9dabf87363a50e7de85ebcf68bc8f114e82a01bfe827a33aa",
        "LFR_N100000_ad5_mc50_mu0.3.json.gz": "aa45e07b74aa57b89c17b3978b33a7c5cf36dd1e1110bc94ee563c4325f00879",
        "LFR_N100000_ad5_mc50_mu0.4.json.gz": "60a7b1392d051a4773bfff428014e1b24111588a770dc600c7d251097c447ee3",
        "LFR_N100000_ad5_mc50_mu0.5.json.gz": "c38a19b512ad44d308ebaa76051259f69d35e8289b3724d0c01e84db51d8802c",
        "LFR_N100000_ad5_mc50_mu0.6.json.gz": "08ef0a2cd65ad063b7ec54c4d38f3b7e531bbc1a18a9d9e49d68121b9f0160e1",
        "LFR_N100000_ad5_mc50_mu0.7.json.gz": "9338497d042b97d128a1df96a631fd3ad15c104469ab5c8a7e425f6a11e75e18",
        "LFR_N100000_ad5_mc50_mu0.8.json.gz": "342062125e4c102b5e0a5e7fc9217dc68c39c002f27da6364a4f48f3fee11434",
        "LFR_N100000_ad5_mc50_mu0.9.json.gz": "ef262d3f1944cbb4f7c5863ede9feba21d1c15e87901fbb0070edb85e8a56b0a",
        "LFR_N10000_ad5_mc50_mu0.1.json.gz": "8c363cdc0a4816b24e7a5e6adbbfa0b398414d76d13806454955b1244f1b1fe0",
        "LFR_N10000_ad5_mc50_mu0.2.json.gz": "f9c6b1c2345fb18ded37ded5d9348004db9aacda681f8b4551253d6962fec54d",
        "LFR_N10000_ad5_mc50_mu0.3.json.gz": "43104baaf276c034b597e7ca79fcb3224828ff942fa93178e209e8dc0853cbcd",
        "LFR_N10000_ad5_mc50_mu0.4.json.gz": "b0eacf94b9b74a9fe405b3b2f017c4b04a1403f848238dda19c20cb67bdbbd36",
        "LFR_N10000_ad5_mc50_mu0.5.json.gz": "13d0d4a27708ea6c19244ce21d34b07d8c5338d32e4e40bb7af1bc8ae0b90cf8",
        "LFR_N10000_ad5_mc50_mu0.6.json.gz": "7f3b760490832644bfa05fd901ede1c799dfa754950ab7a4143871f2c042ca59",
        "LFR_N10000_ad5_mc50_mu0.7.json.gz": "50082826be9e123f4ebc083dd4449eb327fe0e9726a356839bd4be03e7e5f0ce",
        "LFR_N10000_ad5_mc50_mu0.8.json.gz": "6dba81412195f54f58a18f51e095918ddffc80c5feb5b91e7f3abbbe41f886c7",
        "LFR_N10000_ad5_mc50_mu0.9.json.gz": "6447a14df7c4d9e8d4d9c8670178428066e3ee7b0ac0359d733a000d2adb4983",
        "LFR_N1000_ad5_mc50_mu0.1.json.gz": "07e297bc63793c297076e15587196f6311a583e32d6fa58469c88471062ab715",
        "LFR_N1000_ad5_mc50_mu0.2.json.gz": "8d75a29a87d075b1460fcba657bdd2edf10dfca96a3ea240cc6f1c266cbb914c",
        "LFR_N1000_ad5_mc50_mu0.3.json.gz": "e41f900abce6352f1ea8c4e21c8ef27456d397d2af0e4a15c016256971c3dbea",
        "LFR_N1000_ad5_mc50_mu0.4.json.gz": "5e13918acb890a238eff549a5153da720f8d88ef295d5314777fc035e76e2cdf",
        "LFR_N1000_ad5_mc50_mu0.5.json.gz": "9203c87763aca467cfb176f5cab5a2b09f02dc70bdd139822368155ba7589505",
        "LFR_N1000_ad5_mc50_mu0.6.json.gz": "a6682eb003ac8091975a99929bbdcfdb610bedb541ea58f01e4472fedb6329e1",
        "LFR_N1000_ad5_mc50_mu0.7.json.gz": "cf15587099eecf7fb567b2f3796216c48b32812e9d3dbf9898570b730fee9e47",
        "LFR_N1000_ad5_mc50_mu0.8.json.gz": "9dd4ddd0cfe9083d8a6864ecc590b75423d66f45629b3f3aeb2edb264b7adbab",
        "LFR_N1000_ad5_mc50_mu0.9.json.gz": "ec1b36bb6e761dfb901e68a6be54b873914de669ced44b3ba2ceaf3a9fbc2910",
        "LFR_N50000_ad5_mc50_mu0.1.json.gz": "8f93424c0c678e290a15e3d1e1d9830d6a4f093f65808b25b65cec6ec4746f86",
        "LFR_N50000_ad5_mc50_mu0.2.json.gz": "696ee1368acf4d4437a23fea8c7f96b651e34f6e3d4936865982eea68d4b1234",
        "LFR_N50000_ad5_mc50_mu0.3.json.gz": "dd032117d5baf699ba09a5d612290f7c5b4562103f3688ce08647509667f1016",
        "LFR_N50000_ad5_mc50_mu0.4.json.gz": "cbe54f699af50c6059a5d29391f7a9397753731ad51a91c1f500a86fdca04130",
        "LFR_N50000_ad5_mc50_mu0.5.json.gz": "d71f92c8425f3af01b041d2bd2549a8863e8f294e5252b74afd7acbcbafd9bfd",
        "LFR_N50000_ad5_mc50_mu0.6.json.gz": "999286832cb585b98a651ea5d5008bc6194f052326bd24cac2ae73b78dfb803c",
        "LFR_N50000_ad5_mc50_mu0.7.json.gz": "9d129ce9ba90b46ea68617485418ea535d97dd24cad9fd566b7e2c36d44f210c",
        "LFR_N50000_ad5_mc50_mu0.8.json.gz": "20989d85b58818b799b526e1a40c742cd19dfd593fe94ca0f0dba54dc9a86b99",
        "LFR_N50000_ad5_mc50_mu0.9.json.gz": "dd75836ff29cbeef130889cbbf1725af09e6e7a1e8038af106bdcadd8628a63c",
        "LFR_N5000_ad5_mc50_mu0.1.json.gz": "cc4ea02e10184eee588978a79a8aa2b9f0805090ba1c3d4b8c7224c242f96517",
        "LFR_N5000_ad5_mc50_mu0.2.json.gz": "9598c8e20857951edfc41cc5b42f841d01b423982552196440fc37284c0191d3",
        "LFR_N5000_ad5_mc50_mu0.3.json.gz": "a6f04d0a483775ca812004ab435ffb4df992152b446d432d3ba6eedaa76e278c",
        "LFR_N5000_ad5_mc50_mu0.4.json.gz": "b678994df15c884cb3f0ce4d441039aff4f48c30acfc08cf6d61cbe03366e4ce",
        "LFR_N5000_ad5_mc50_mu0.5.json.gz": "b0281d2c36562743d582b7021160263a5f9db31f4970c698a332b063cb4c6d7c",
        "LFR_N5000_ad5_mc50_mu0.6.json.gz": "4c1e95a89e27676b55e851992d724a9248ccf01994ac0a83415b25ac17607e2d",
        "LFR_N5000_ad5_mc50_mu0.7.json.gz": "fc211d363947a41b028ccd259a7a1f02933abd7a5a458874c9505faa9d24d662",
        "LFR_N5000_ad5_mc50_mu0.8.json.gz": "e09ca4318e4e63b10e3c27837221c17e5d904a4c5e954c698a0ba98cc419bd47",
        "LFR_N5000_ad5_mc50_mu0.9.json.gz": "5287d16543efcac9f5748744f690eadeb61d46b14cb95569b104308935a74583",
    },
)


def available_networks() -> list:
    """
    List the remotely available network datasets.

    :return: list of network names

    :Example:

    >>> from cdlib import datasets
    >>> graph_name_list = datasets.available_networks()

    """
    return [x.split(".csv")[0] for x in __networks.registry.keys()]


def available_ground_truths() -> list:
    """
    List the remotely available network ground truth datasets.

    :return: list of network names

    :Example:

    >>> from cdlib import datasets
    >>> graph_name_list = datasets.available_ground_truths()

    """
    return [x.split(".json")[0] for x in __ground_truths.registry.keys()]


def fetch_network_data(
    net_name: str = "karate_club", net_type: str = "igraph"
) -> object:
    """
    Load the required network from the remote repository

    :param net_name: network name
    :param net_type: desired graph object among "networkx" and "igraph". Default, igraph.
    :return: a graph object

    :Example:

    >>> from cdlib import datasets
    >>> G = datasets.fetch_network_data(net_name="karate_club", net_type="igraph")

    """

    download = pooch.HTTPDownloader(progressbar=True)
    fname = __networks.fetch(
        f"{net_name}.csv.gz", processor=pooch.Decompress(), downloader=download
    )

    if net_type == "networkx":
        g = nx.Graph()
        with open(fname) as f:
            for line in f:
                line = line.replace(" ", "\t").split("\t")
                g.add_edge(int(line[0]), int(line[1]))
    else:
        if ig is None:
            raise ModuleNotFoundError(
                "Optional dependency not satisfied: install python-igraph to use the selected "
                "feature."
            )

        edges = []
        with open(fname) as f:
            for line in f:
                line = line.replace(" ", "\t").split("\t")
                edges.append((int(line[0]), int(line[1])))
        g = ig.Graph.TupleList(edges)

    return g


def fetch_ground_truth_data(
    net_name: str = "karate_club", graph: object = None
) -> object:
    """
    Load the required ground truth clustering from the remote repository

    :param net_name: network name
    :param graph: the graph object associated to the ground truth (optional)
    :return: a NodeClustering object

    :Example:

    >>> from cdlib import datasets
    >>> gt_coms = datasets.fetch_network_data(fetch_ground_truth_data="karate_club")

    """

    download = pooch.HTTPDownloader(progressbar=True)
    fname = __ground_truths.fetch(
        f"{net_name}.json.gz", processor=pooch.Decompress(), downloader=download
    )
    gt = read_community_json(fname)
    if graph is not None:
        gt.graph = graph
    return gt


def fetch_network_ground_truth(
    net_name: str = "karate_club", net_type: str = "igraph"
) -> [object, object]:
    """
    Load the required network, along with its ground truth partition, from the remote repository.

    :param net_name: network name
    :param net_type: desired graph object among "networkx" and "igraph". Default, igraph.
    :return: a tuple of (graph_object, NodeClustering)

    :Example:

    >>> from cdlib import datasets
    >>> G, gt_coms = datasets.fetch_network_ground_truth(fetch_ground_truth_data="karate_club", net_type="igraph")

    """

    if (
        net_name not in available_networks()
        or net_name not in available_ground_truths()
    ):
        raise ValueError(f"{net_name} is not present in the remote repository")

    g = fetch_network_data(net_name, net_type)
    gt = fetch_ground_truth_data(net_name, g)
    return g, gt
