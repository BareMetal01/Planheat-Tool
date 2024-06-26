import logging
import networkx as nx
import geonetworkx as gnx
from .. import config
from ..graph_building_base import GraphBuilder


class MDGraphBuilder(GraphBuilder):
    """
    This objects defines steps to creates an optimization cost corresponding to the Steiner Tree Problem (STP).
    """

    def __init__(self, **kwargs):
        super(MDGraphBuilder, self).__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

        self.existing_network_merged_graph = None

    def generate_graph(self):
        """Main method that runs all steps one by one."""
        self.check_is_ready()
        self.logger.info("Start optimization graph building for Steiner Tree Problem")
        self.import_street_graph_from_open_street_map()
        if self.initial_street_graph is None:
            self.initial_street_graph = self.street_graph.copy()
        self.street_graph = self.initial_street_graph.copy()
        self.remove_streets_to_exclude()
        self.merge_existing_network()
        node_filter = lambda n: n not in self.old_network_buildings
        buildings_filter = lambda b: b in self.marked_buildings
        self.add_buildings(self.existing_network_merged_graph, node_filter=node_filter,
                           buildings_filter=buildings_filter)
        self.finalize_optimization_graph()
        self.logger.info("Optimization graph building end")

    def merge_existing_network(self):
        """Merge an existing network to the study area street graph."""
        if len(self.old_network_buildings) > 0 or len(self.old_network_streets) > 0:
            # Compose the street graph with the existing network
            if self.old_network_graph is None:
                self.generate_old_network_graph()
            merged_graph = nx.compose(self.street_graph, self.old_network_graph)
            merged_graph.crs = self.street_graph.crs
            # Remove the edges (the original ones) that have been split to add a building
            edges_to_remove = set()
            for e in self.old_network_streets:
                if e not in self.old_network_graph.edges:
                    edges_to_remove.add(e)
            merged_graph.remove_edges_from(edges_to_remove)
            self.existing_network_merged_graph = merged_graph
        else:
            self.existing_network_merged_graph = self.street_graph

    def finalize_optimization_graph(self):
        """Finalize the optimization graph by setting it up and defining a cost structure on edges."""
        self.logger.info("Finalization of the optimization graph start")
        self.optimization_graph = self.building_merged_graph
        # Fill geometry attribute
        gnx.fill_edges_missing_geometry_attributes(self.optimization_graph)
        # Set length
        gnx.fill_length_attribute(self.optimization_graph, attribute_name=config.EDGE_LENGTH_KEY, only_missing=True)
        # fill cost
        self.fill_edges_cost_attributes(self.optimization_graph)
        self.optimization_graph.name = "optimization_graph"
        # gnx.export_graph_as_shape_file(self.optimization_graph, r"D:\projets\PlanHeat\PlanHeatGnxWork\tmp",
        #  fiona_cast=True)
        self.logger.info("Finalization of the optimization graph end")

    def finalize_old_network_graph(self):
        pass


if __name__ == "__main__":
    import os

    data_dir = "../../data/Antwerp_01"
    results_dir = "../../optimizer/manual_design/results/"
    # district area
    district_shape = os.path.join(data_dir, "antwerp_01_shape.shp")
    # buildings
    buildings_path = os.path.join(data_dir, "antwerp_01_buildings.shp")
    # Set graph builder
    self = MDGraphBuilder()

    self.logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s')
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    self.logger.addHandler(stream_handler)

    self.district = district_shape
    self.buildings_file_path = buildings_path
    """
    self.old_network_buildings = {'B_0', 'B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_11',
                                  'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19'}
    self.old_network_streets = {('1904622081', '223825946'), ('1904622081', '1904622108'), ('1904622081', '1890438025'),
                                ('286307329', '4761993311'), ('286307329', '252944688'), ('286307329', '223850989'),
                                ('217765891', '208521768'), ('217765891', '223863059'), ('217765891', '206137273'),
                                ('286307845', '4761993305'), ('286307845', '5476211921'), ('286307845', '5476211926'),
                                ('286307851', '5476211926'), ('286307851', '223814924'), ('286307851', '223817526'),
                                ('205456395', '26522181'), ('205456395', '206105851'), ('205456395', '206137291'),
                                ('280716814', '195585877'), ('280716814', '195585878'), ('280716814', '280716799'),
                                ('220725268', '220729562'), ('220725268', '221080812'), ('220725268', '220727141'),
                                ('220725268', '220724162'), ('247680534', '247680539'), ('247680534', '197759280'),
                                ('247680534', '217751487'), ('206105624', '217762861'), ('206105624', '223854853'),
                                ('206105624', '223863059'), ('206105624', '206137273'), ('223825946', '5261750890'),
                                ('223825946', '196191113'), ('247680539', '217746618'), ('247680539', '217751487'),
                                ('281482268', '197760509'), ('281482268', '197759325'), ('281482268', '197760506'),
                                ('1904622108', '203379820'), ('1904622108', '4241948344'), ('1904622108', '26152754'),
                                ('217759774', '223861924'), ('217759774', '217762101'), ('217759774', '206137291'),
                                ('1899087902', '197760509'), ('1899087902', '290387192'), ('1899087902', '290383706'),
                                ('370800156', '4376546422'), ('370800156', '370799842')}
    """
    # self.old_network_buildings = {'5326305.0', '3736216.0', '3736557.0', '3736620.0', '5321162.0', '5321174.0',
    #                               '3736564.0', '5526900.0', '3736736.0'}
    # self.old_network_streets = {('321429208', '290387192'), ('1899087902', '290387192'), ('223842739', '290387192'),
    #                             ('321429208', '197759289'), ('197759280', '197759289'), ('290384108', '290387192')}

    self.marked_buildings = {'5321187.0', '5321165.0', '5321099.0', '5326321.0', '5321218.0', '5321219.0', '3737065.0',
                             '5326308.0', '3736531.0', '5326263.0', '5321207.0', '3737043.0', '3736950.0', '3737006.0',
                             '5321186.0', '5321152.0', '3736521.0', '3736965.0', '5321145.0', '3736949.0'}
    self.excluded_streets = {('4425526631', '5868786923'), ('5344904184', '276785748'), ('4935695474', '4935695470'),
                             ('4425526631', '4379660337'), ('221471611', '221470334'), ('195585963', '205458506'),
                             ('4241948344', '1904622108'), ('205466281', '276785748'), ('1904589710', '220727141'),
                             ('370626895', '370800156'), ('4425526639', '4379647009'), ('220734940', '206137275'),
                             ('4241948344', '4376546422'), ('220706669', '4878471829'), ('237768189', '205465493'),
                             ('655344485', '655344484'), ('370626895', '370628404'), ('197762183', '27307067'),
                             ('196191118', '195585878'), ('4241948312', '4379647009'), ('220729562', '220731035'),
                             ('220739268', '203379815'), ('217754032', '217754302'), ('217746618', '217746617'),
                             ('260791188', '195752799'), ('4246205430', '220724162'), ('26522010', '27307038'),
                             ('220740583', '252944688'), ('1904589682', '220731035'), ('221472730', '221481097'),
                             ('4425526639', '4814057233'), ('195585958', '655344484'), ('655344486', '655344483'),
                             ('370795115', '4379660340'), ('203379814', '221110341'), ('195585960', '195585959'),
                             ('1904589682', '1904589676'), ('280716799', '280716814'), ('4246205430', '206137435'),
                             ('197762184', '247680546'), ('1904589718', '549920844'), ('4376546422', '4379660337'),
                             ('205466985', '237770308'), ('223863059', '217765891'), ('221080812', '220725268'),
                             ('237768189', '276785748'), ('26522084', '205465493'), ('223861924', '208521768'),
                             ('221472730', '206137273'), ('220734940', '221472730'), ('208521768', '217765891'),
                             ('26522035', '237774592'), ('260791188', '26152756'), ('4886011299', '4890147492'),
                             ('206105851', '206137046'), ('4241948319', '4241948312'), ('549920844', '221110341'),
                             ('252944688', '203379815'), ('370628404', '26152755'), ('206105528', '206137046'),
                             ('2621468320', '27307035'), ('195585960', '195614287'), ('217762101', '217762857'),
                             ('370626895', '4379660340'), ('195585947', '655344483'), ('260791138', '280717117'),
                             ('260791138', '26152755'), ('195585963', '195585960'), ('1904589676', '221476998'),
                             ('6125781022', '4379647011'), ('208521490', '208521768'), ('26522035', '2621468320'),
                             ('195752799', '195585880'), ('197759332', '195614287'), ('195732891', '195585880'),
                             ('217751485', '247680545'), ('220724162', '4241948310'), ('205465493', '197762186'),
                             ('220724162', '220725268'), ('221472727', '221481097'), ('195585957', '59119524'),
                             ('3255985140', '1085435886'), ('203379820', '1904622108'), ('280716705', '370799842'),
                             ('220736868', '221482135'), ('280716799', '196191116'), ('59119524', '247370288'),
                             ('237768189', '237770308'), ('4810665864', '221470334'), ('4241948335', '4241948328'),
                             ('205468416', '237770308'), ('217753550', '217754032'), ('203379820', '1086546991'),
                             ('370799842', '370800156'), ('197759332', '197759328'), ('195732891', '195585879'),
                             ('220734940', '221482135'), ('220740583', '4860674446'), ('195585957', '195590767'),
                             ('195732891', '370631286'), ('206105851', '205456395'), ('2616411596', '26152756'),
                             ('220727141', '220725268'), ('280716705', '196191116'), ('217750806', '197762186'),
                             ('26152754', '4376546422'), ('4425526631', '4814057221'), ('237771010', '616237154'),
                             ('205465493', '205466985'), ('220727144', '220731035'), ('26522084', '27307067'),
                             ('4379660340', '4379660337'), ('217753550', '217760949'), ('195585947', '195585926'),
                             ('4144373849', '205458506'), ('205466985', '237771010'), ('5344904184', '217750806'),
                             ('237774592', '616236167'), ('197764997', '197759332'), ('307694556', '247370540'),
                             ('208521490', '206137046'), ('220740583', '220739268'), ('195585926', '5512180516'),
                             ('220735931', '220736868'), ('27307067', '27307035'), ('2000785683', '220689674'),
                             ('252944688', '286307329'), ('195585967', '280717117'), ('280717118', '370799842'),
                             ('220689674', '206137435'), ('286307135', '286307070'), ('223850989', '203379820'),
                             ('221471611', '286307070'), ('221476998', '221470334'), ('206105528', '220706669'),
                             ('1904622108', '1904622081'), ('195585877', '280716814'), ('220735931', '221470334'),
                             ('196191113', '26152754'), ('26522010', '27307043'), ('237771010', '27307044'),
                             ('237759956', '4886011299'), ('195585959', '195585958'), ('4814057235', '4241948311'),
                             ('3255985140', '197762183'), ('217762101', '217760949'), ('247370540', '4379660337'),
                             ('195590767', '195614287'), ('217746616', '26522181'), ('217754032', '217746617'),
                             ('237759956', '247370288'), ('221087464', '220729562'), ('1904589718', '1904589710'),
                             ('247680597', '26522181'), ('206137291', '217759774'), ('2621468320', '27307043'),
                             ('26522035', '27307044'), ('217762101', '217759774'), ('4878471830', '4878471829'),
                             ('206137291', '205456395'), ('4379647011', '4379647010'), ('220735931', '4935695470'),
                             ('217746617', '217746616'), ('220729562', '220725268'), ('224405647', '27307039'),
                             ('217755947', '217760949'), ('307694556', '26152756'), ('4814057235', '4379647011'),
                             ('26152754', '1904622108'), ('59119524', '195585947'), ('2893378503', '4144373849'),
                             ('280717118', '26152755'), ('307694556', '370628404'), ('223814924', '221476998'),
                             ('4241948335', '6125781022'), ('195585963', '195585878'), ('195585878', '280716814'),
                             ('260791138', '370631286'), ('247680597', '247680546'), ('655344485', '195585879'),
                             ('195585877', '4144373849'), ('195585926', '655344486'), ('4810665864', '286307070'),
                             ('280716799', '196191118'), ('280716705', '26152754'), ('59119529', '195614287'),
                             ('59119562', '197764997'), ('1904589710', '220727144'), ('4241948342', '4241948338'),
                             ('206137291', '217754032'), ('217746616', '247680545'), ('217753550', '217745866'),
                             ('616236167', '616237154'), ('220718913', '220689674'), ('203379815', '203379814'),
                             ('220718913', '4878471829'), ('26522010', '224405647'), ('27307044', '27307035'),
                             ('195585958', '195585957'), ('206137291', '208521490'), ('59119529', '59119524'),
                             ('4379647011', '4379647009'), ('206137273', '217765891'), ('260791188', '260791113'),
                             ('220718913', '221080812'), ('2616411561', '195585880'), ('1904589682', '4935695470'),
                             ('4241948312', '4241948311'), ('4425526639', '5868786923'), ('237766988', '616236167'),
                             ('2000785683', '221080812'), ('195585879', '280717117'), ('4241948311', '4241954927'),
                             ('370628404', '370631289'), ('195585967', '370799842'), ('221482135', '221481097'),
                             ('280717118', '280717117'), ('195585967', '196191118'), ('220740583', '221476998'),
                             ('4379660337', '4379647010'), ('237766988', '616237154'), ('247680597', '27307039'),
                             ('221472727', '286307135'), ('220727144', '220739268'), ('4886011299', '237766988'),
                             ('26522084', '27307035'), ('26522181', '205456395'), ('220727141', '549920844'),
                             ('195585959', '195590767'), ('655344486', '655344485'), ('195585879', '195585878'),
                             ('206137275', '206105528'), ('4241948327', '6125781022'), ('4241948310', '4241954927'),
                             ('260791113', '370631289'), ('2616411596', '247370540'), ('205468416', '247370288'),
                             ('205458507', '205458506'), ('223850989', '223852206'), ('4814057235', '4241948344'),
                             ('4376546422', '370800156'), ('224405647', '197762183'), ('4241948338', '4241948335'),
                             ('4241948310', '221110341'), ('247680546', '247680545'), ('1635630583', '197764997'),
                             ('221472727', '4810665864'), ('217751485', '217750806'), ('1904589676', '220727144'),
                             ('223850989', '286307329'), ('1904589718', '203379814'), ('197762184', '197762186'),
                             ('4241954927', '6125781022'), ('2616411561', '5512180516'), ('4241948344', '4379647010'),
                             ('27307038', '27307039'), ('220736868', '4935695468'), ('370631289', '370631286'),
                             ('655344484', '655344483'), ('206137275', '208521490'), ('237759956', '237770308'),
                             ('3255985140', '197762184'), ('4246205430', '2000785683'), ('205466281', '59119529'),
                             ('221599316', '205466281'), ('205468416', '59119529')}
    self.import_street_graph_from_open_street_map()
    self.remove_streets_to_exclude()
    self.generate_old_network_graph()

    self.generate_graph()
    # gnx.export_graph_as_shape_file(self.optimization_graph, results_dir, fiona_cast=True)
    self.old_network_graph.name = "old_network_graph"
    # gnx.export_graph_as_shape_file(self.old_network_graph, results_dir, fiona_cast=True)
    gnx.write_gpickle(self.optimization_graph, os.path.join(results_dir, "optimization_graph.gpickle"))
    gnx.write_gpickle(self.old_network_graph, os.path.join(results_dir, "old_network_graph.gpickle"))
