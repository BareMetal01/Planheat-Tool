# -*- coding: utf-8 -*-
"""
/***************************************************************************
 PlanHeatDPM
                                 A QGIS plugin
 District Planninng Module
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2018-09-04
        git sha              : $Format:%H$
        copyright            : (C) 2018 by andbs
        email                : andbs@rina.org
 ***************************************************************************/
"""
# Import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTreeWidgetItem

# Import numerical items
from pandas import DataFrame, MultiIndex
from numpy import average

# Import qgis main libraries
from qgis.core import *


class Building(QTreeWidgetItem):
    """
    Class that inherits the QTreeWidgetItem whilst accepting the technology
    dropped
    """

    def __init__(self, feature: QgsFeature):
        """
        Initalise the building class instance
        :param feature:
        """
        QTreeWidgetItem.__init__(
            self, [str(feature.id())], type=QTreeWidgetItem.Type
        )

        self.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        self.dhcn = False
        self.feature = feature
        self.building_id = self.feature.attribute(
            self.feature.fieldNameIndex('BuildingID')
        )
        self.setText(0, str(feature.id()))
        cool_dem = self.feature.attribute('MaxCoolDem')
        heat_dem = self.feature.attribute('MaxHeatDem')
        dhw_dem = self.feature.attribute('MaxDHWDem')
        dhwh_dem = self.feature.attribute('MaxHeatDem') + \
                   self.feature.attribute('MaxDHWDem')
        self.cooling = QTreeWidgetItem(
            ['Cooling'],
            type=QTreeWidgetItem.Type
        )
        self.cooling.setData(1, Qt.DisplayRole, cool_dem)

        self.heating = QTreeWidgetItem(
            ['Heating'],
            type=QTreeWidgetItem.Type
        )
        self.heating.setData(1, Qt.DisplayRole, heat_dem)

        self.dhw = QTreeWidgetItem(
            ['DHW'],
            type=QTreeWidgetItem.Type
        )
        self.dhw.setData(1, Qt.DisplayRole, dhw_dem)

        self.dhwh = QTreeWidgetItem(
            ['DHW and Heating'],
            type=QTreeWidgetItem.Type
        )
        self.dhwh.setData(1, Qt.DisplayRole, dhwh_dem)

        self.addChildren([
            self.cooling,
            self.heating,
            self.dhw,
            # self.dhwh
        ])

        self.dhn = None
        self.dcn = None
        building_index = MultiIndex.from_product(
            [
                ['Cooling', 'Heating', 'DHW', 'DHW_Heating'],
                ['-'],
                ['-']
            ],
            names=['Demand', 'Source', 'Technology']
        )
        building_columns = [
            'Peak Demand',
            'Capacity',
            'Efficiency',
            'Technical Minimum',
            'Ramp Up',
            'Ramp Down',
            'Fixed Cost',
            'Variable Cost',
            'Fuel Cost',
            'DHN',
            'DCN'
        ]
        self.building_data = DataFrame(
            columns=building_columns,
            index=building_index
        )

    def toggle_dhn(self, dhn_id: str):
        """
        Connect/disconnect the building to the DHN
        :param dhn_id: the name of the DHN
        :return:
        """
        if self.dhn:
            self.dhn = None
            self.setText(13, '')
            self.heating.setDisabled(False)
            self.dhw.setDisabled(False)
            self.dhwh.setDisabled(False)
        else:
            self.dhn = dhn_id
            self.setText(13, dhn_id)
            self.heating.setDisabled(True)
            self.dhw.setDisabled(True)
            self.dhwh.setDisabled(True)

    def toggle_dcn(self, dcn_id: str):
        """
        Connect/disconnect the building to the DCN
        :param dcn_id: the name of the DCN
        :return:
        """
        if self.dcn:
            self.dcn = None
            self.setText(12, '')
            self.cooling.setDisabled(False)
        else:
            self.dcn = dcn_id
            self.setText(12, dcn_id)
            self.cooling.setDisabled(True)

    def update_data(self):
        """
        Update the heating total output and efficiency
        :return:
        """
        return