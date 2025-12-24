import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Switch, TextInput, Alert, Platform, Image } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import Icon from '../components/Icon';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Picker } from '@react-native-picker/picker';
import CoeLogo from '../assets/images/Coe.png';
import SRMLogo from '../assets/images/SRM_logo.png';
import TataSteelLogo from '../assets/images/tata_steel.png';

const LANGUAGES = [
  { label: 'English', value: 'en' },
  { label: 'Hindi', value: 'hi' },
  { label: 'Bengali', value: 'bn' },
  { label: 'Tamil', value: 'ta' },
  { label: 'Telugu', value: 'te' },
];

const MEASUREMENT_UNITS = [
  { label: 'Millimeters (mm)', value: 'mm' },
  { label: 'Centimeters (cm)', value: 'cm' },
  { label: 'Inches (in)', value: 'in' },
];

const CAMERA_QUALITIES = [
  { label: 'High Quality', value: 'high' },
  { label: 'Medium Quality', value: 'medium' },
  { label: 'Low Quality', value: 'low' },
];

const EXPORT_FORMATS = [
  { label: 'PDF Report', value: 'pdf' },
  { label: 'CSV Data', value: 'csv' },
  { label: 'Both', value: 'both' },
];

const SettingsScreen = () => {
  const navigation = useNavigation();
  const [notifications, setNotifications] = useState(true);
  const [language, setLanguage] = useState('en');
  const [measurementUnit, setMeasurementUnit] = useState('mm');
  const [cameraQuality, setCameraQuality] = useState('high');
  const [autoSaveResults, setAutoSaveResults] = useState(true);
  const [calibrationMode, setCalibrationMode] = useState(false);
  const [exportFormat, setExportFormat] = useState('pdf');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load settings from AsyncStorage
    const loadSettings = async () => {
      try {
        const notif = await AsyncStorage.getItem('notifications');
        const lang = await AsyncStorage.getItem('language');
        const unit = await AsyncStorage.getItem('measurementUnit');
        const quality = await AsyncStorage.getItem('cameraQuality');
        const autoSave = await AsyncStorage.getItem('autoSaveResults');
        const calibration = await AsyncStorage.getItem('calibrationMode');
        const exportFmt = await AsyncStorage.getItem('exportFormat');
        
        if (notif !== null) setNotifications(notif === 'true');
        if (lang !== null) setLanguage(lang);
        if (unit !== null) setMeasurementUnit(unit);
        if (quality !== null) setCameraQuality(quality);
        if (autoSave !== null) setAutoSaveResults(autoSave === 'true');
        if (calibration !== null) setCalibrationMode(calibration === 'true');
        if (exportFmt !== null) setExportFormat(exportFmt);
      } catch (e) {}
      setLoading(false);
    };
    loadSettings();
  }, []);

  const saveSetting = async (key, value) => {
    try {
      await AsyncStorage.setItem(key, value.toString());
    } catch (e) {}
  };

  const handleNotifications = (val) => {
    setNotifications(val);
    saveSetting('notifications', val);
  };
  
  const handleLanguage = (val) => {
    setLanguage(val);
    saveSetting('language', val);
  };
  
  const handleMeasurementUnit = (val) => {
    setMeasurementUnit(val);
    saveSetting('measurementUnit', val);
  };
  
  const handleCameraQuality = (val) => {
    setCameraQuality(val);
    saveSetting('cameraQuality', val);
  };
  
  const handleAutoSaveResults = (val) => {
    setAutoSaveResults(val);
    saveSetting('autoSaveResults', val);
  };
  
  const handleCalibrationMode = (val) => {
    setCalibrationMode(val);
    saveSetting('calibrationMode', val);
  };
  
  const handleExportFormat = (val) => {
    setExportFormat(val);
    saveSetting('exportFormat', val);
  };
  
  const handleLogout = () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Logout',
          style: 'destructive',
          onPress: async () => {
            try {
              // Clear any authentication data
              await AsyncStorage.removeItem('isLoggedIn');
              await AsyncStorage.removeItem('userToken');
              // Navigate back to login screen
              navigation.reset({
                index: 0,
                routes: [{ name: 'Login' }],
              });
            } catch (error) {
              console.error('Logout error:', error);
              // Still navigate to login even if clearing fails
              navigation.reset({
                index: 0,
                routes: [{ name: 'Login' }],
              });
            }
          },
        },
      ]
    );
  };
  
  const handleClearData = () => {
    Alert.alert(
      'Clear App Data',
      'Are you sure you want to clear all app data? This cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            await AsyncStorage.clear();
            setNotifications(true);
            setLanguage('en');
            setMeasurementUnit('mm');
            setCameraQuality('high');
            setAutoSaveResults(true);
            setCalibrationMode(false);
            setExportFormat('pdf');
            Alert.alert('Data Cleared', 'All app data has been cleared.');
          },
        },
      ]
    );
  };

  if (loading) {
    return (
      <LinearGradient colors={['#0A0A0A', '#1A1A2E', '#16213E', '#0F3460']} style={styles.container}>
        <Text style={{ color: '#fff', textAlign: 'center', marginTop: 100 }}>Loading settings...</Text>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient colors={['#0A0A0A', '#1A1A2E', '#16213E', '#0F3460']} style={styles.container}>
      <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
        <Icon name="arrow-back" size={28} color="#00D4FF" />
      </TouchableOpacity>
      <View style={styles.content}>
        <Text style={styles.title}>Settings</Text>
        
        {/* Notifications Toggle */}
        <View style={styles.settingRow}>
          <View style={styles.settingLabelWrap}>
            <Icon name="notifications-active" size={24} color="#00D4FF" style={{ marginRight: 12 }} />
            <Text style={styles.settingLabel}>Notifications</Text>
          </View>
          <Switch
            value={notifications}
            onValueChange={handleNotifications}
            thumbColor={notifications ? '#00D4FF' : '#888'}
            trackColor={{ false: '#444', true: '#16213E' }}
          />
        </View>
        
        {/* Language Picker */}
        <View style={styles.settingRow}>
          <View style={styles.settingLabelWrap}>
            <Icon name="language" size={24} color="#00D4FF" style={{ marginRight: 12 }} />
            <Text style={styles.settingLabel}>Language</Text>
          </View>
          <View style={styles.pickerWrap}>
            <Picker
              selectedValue={language}
              style={styles.picker}
              dropdownIconColor="#00D4FF"
              onValueChange={handleLanguage}
              mode={Platform.OS === 'ios' ? 'dialog' : 'dropdown'}
            >
              {LANGUAGES.map((lang) => (
                <Picker.Item key={lang.value} label={lang.label} value={lang.value} />
              ))}
            </Picker>
          </View>
        </View>
        
        {/* Measurement Units */}
        <View style={styles.settingRow}>
          <View style={styles.settingLabelWrap}>
            <Icon name="straighten" size={24} color="#00D4FF" style={{ marginRight: 12 }} />
            <Text style={styles.settingLabel}>Measurement Units</Text>
          </View>
          <View style={styles.pickerWrap}>
            <Picker
              selectedValue={measurementUnit}
              style={styles.picker}
              dropdownIconColor="#00D4FF"
              onValueChange={handleMeasurementUnit}
              mode={Platform.OS === 'ios' ? 'dialog' : 'dropdown'}
            >
              {MEASUREMENT_UNITS.map((unit) => (
                <Picker.Item key={unit.value} label={unit.label} value={unit.value} />
              ))}
            </Picker>
          </View>
        </View>
        
        {/* Camera Quality */}
        <View style={styles.settingRow}>
          <View style={styles.settingLabelWrap}>
            <Icon name="camera-alt" size={24} color="#00D4FF" style={{ marginRight: 12 }} />
            <Text style={styles.settingLabel}>Camera Quality</Text>
          </View>
          <View style={styles.pickerWrap}>
            <Picker
              selectedValue={cameraQuality}
              style={styles.picker}
              dropdownIconColor="#00D4FF"
              onValueChange={handleCameraQuality}
              mode={Platform.OS === 'ios' ? 'dialog' : 'dropdown'}
            >
              {CAMERA_QUALITIES.map((quality) => (
                <Picker.Item key={quality.value} label={quality.label} value={quality.value} />
              ))}
            </Picker>
          </View>
        </View>
        
        {/* Auto-save Results */}
        <View style={styles.settingRow}>
          <View style={styles.settingLabelWrap}>
            <Icon name="save" size={24} color="#00D4FF" style={{ marginRight: 12 }} />
            <Text style={styles.settingLabel}>Auto-save Results</Text>
          </View>
          <Switch
            value={autoSaveResults}
            onValueChange={handleAutoSaveResults}
            thumbColor={autoSaveResults ? '#00D4FF' : '#888'}
            trackColor={{ false: '#444', true: '#16213E' }}
          />
        </View>
        
        {/* Calibration Mode */}
        <View style={styles.settingRow}>
          <View style={styles.settingLabelWrap}>
            <Icon name="tune" size={24} color="#00D4FF" style={{ marginRight: 12 }} />
            <Text style={styles.settingLabel}>Calibration Mode</Text>
          </View>
          <Switch
            value={calibrationMode}
            onValueChange={handleCalibrationMode}
            thumbColor={calibrationMode ? '#00D4FF' : '#888'}
            trackColor={{ false: '#444', true: '#16213E' }}
          />
        </View>
        
        {/* Export Format */}
        <View style={styles.settingRow}>
          <View style={styles.settingLabelWrap}>
            <Icon name="file-download" size={24} color="#00D4FF" style={{ marginRight: 12 }} />
            <Text style={styles.settingLabel}>Export Format</Text>
          </View>
          <View style={styles.pickerWrap}>
            <Picker
              selectedValue={exportFormat}
              style={styles.picker}
              dropdownIconColor="#00D4FF"
              onValueChange={handleExportFormat}
              mode={Platform.OS === 'ios' ? 'dialog' : 'dropdown'}
            >
              {EXPORT_FORMATS.map((format) => (
                <Picker.Item key={format.value} label={format.label} value={format.value} />
              ))}
            </Picker>
          </View>
        </View>
        
        {/* Clear Data Button */}
        <TouchableOpacity style={styles.clearButton} onPress={handleClearData}>
          <Icon name="delete-forever" size={22} color="#F44336" style={{ marginRight: 8 }} />
          <Text style={styles.clearButtonText}>Clear App Data</Text>
        </TouchableOpacity>
        
        {/* Logout Button */}
        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <Icon name="logout" size={22} color="#FF6B6B" style={{ marginRight: 8 }} />
          <Text style={styles.logoutButtonText}>Logout</Text>
        </TouchableOpacity>
        
        {/* About Section */}
        <View style={styles.aboutSection}>
          <Icon name="info" size={22} color="#00D4FF" style={{ marginBottom: 6 }} />
          <Text style={styles.aboutTitle}>About</Text>
          <Text style={styles.aboutText}>TMT Quality Testing App v1.0.0{"\n"}Developed by CoE for AgenticTwins, SRMIST</Text>
          <View style={styles.logoRow}>
            <Image 
              source={Platform.OS === 'web' 
                ? { uri: CoeLogo } 
                : CoeLogo
              } 
              style={styles.logoImg} 
              resizeMode="contain"
              onError={(e) => console.log('Coe logo error:', e)}
            />
            <Image 
              source={Platform.OS === 'web'
                ? { uri: SRMLogo }
                : SRMLogo
              } 
              style={styles.logoImg} 
              resizeMode="contain"
              onError={(e) => console.log('SRM logo error:', e)}
            />
            <Image 
              source={Platform.OS === 'web'
                ? { uri: TataSteelLogo }
                : TataSteelLogo
              } 
              style={styles.logoImg} 
              resizeMode="contain"
              onError={(e) => console.log('Tata Steel logo error:', e)}
            />
          </View>
        </View>
      </View>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 60,
  },
  backButton: {
    position: 'absolute',
    top: 60,
    left: 20,
    zIndex: 1,
  },
  content: {
    flex: 1,
    alignItems: 'stretch',
    justifyContent: 'flex-start',
    paddingHorizontal: 32,
    paddingTop: 40,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 32,
    alignSelf: 'center',
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 28,
  },
  settingLabelWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  settingLabel: {
    fontSize: 18,
    color: '#fff',
    fontWeight: '500',
  },
  pickerWrap: {
    minWidth: 120,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(0,212,255,0.2)',
    overflow: 'hidden',
  },
  picker: {
    color: '#00D4FF',
    width: 120,
    height: 40,
  },
  input: {
    minWidth: 160,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(0,212,255,0.2)',
    color: '#00D4FF',
    paddingHorizontal: 10,
    height: 40,
    fontSize: 16,
  },
  clearButton: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'center',
    marginTop: 16,
    marginBottom: 16,
    backgroundColor: 'rgba(255,107,107,0.08)',
    borderRadius: 8,
    paddingHorizontal: 18,
    paddingVertical: 10,
  },
  clearButtonText: {
    color: '#F44336',
    fontSize: 16,
    fontWeight: 'bold',
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'center',
    marginTop: 16,
    marginBottom: 16,
    backgroundColor: 'rgba(255,107,107,0.08)',
    borderRadius: 8,
    paddingHorizontal: 18,
    paddingVertical: 10,
  },
  logoutButtonText: {
    color: '#FF6B6B',
    fontSize: 16,
    fontWeight: 'bold',
  },
  aboutSection: {
    alignItems: 'center',
    marginTop: 24,
    padding: 16,
    backgroundColor: 'rgba(255,255,255,0.04)',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(0,212,255,0.1)',
  },
  aboutTitle: {
    color: '#00D4FF',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  aboutText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 15,
    textAlign: 'center',
  },
  logoRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 18,
    gap: 16,
  },
  logoImg: {
    width: 190,
    height: 190,
    marginHorizontal: 12,
  },
});

export default SettingsScreen; 