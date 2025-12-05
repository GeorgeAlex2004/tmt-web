// ==================== screens/ReportScreen.js ====================
import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
  Dimensions,
  Image,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Icon from 'react-native-vector-icons/MaterialIcons';
import PDFReportGenerator from '../utils/pdfGenerator';
import { BACKEND_URL } from '../config';

const { width } = Dimensions.get('window');

const ReportScreen = ({ route, navigation }) => {
  const { diameter, analysisData } = route.params;
  const [generatingPDF, setGeneratingPDF] = useState(false);
  const [pdfUri, setPdfUri] = useState(null);
  
  const pdfGenerator = new PDFReportGenerator();
  const currentDate = new Date().toLocaleDateString();
  const currentTime = new Date().toLocaleTimeString();
  const testId = `TMT-${Date.now()}`;

  // Extract data from analysisData
  const { 
    verdict, 
    level1, 
    level2, 
    segmented_image_base64, 
    debug_image_base64,
    segmented_image_url,
    debug_image_url
  } = analysisData || {};

  // Helper function to get image URI - prefer base64, fallback to URL
  const getImageUri = (base64Data, urlData) => {
    if (base64Data) {
      // If it's already a data URL, return as is
      if (base64Data.startsWith('data:')) {
        return base64Data;
      }
      // If it's just base64 string, convert to data URL
      return `data:image/png;base64,${base64Data}`;
    }
    if (urlData) {
      return `${BACKEND_URL}${urlData}`;
    }
    return null;
  };

  // Get image URIs
  const segmentedImageUri = getImageUri(segmented_image_base64, segmented_image_url);
  const debugImageUri = getImageUri(debug_image_base64, debug_image_url);

  // Helper function to format labels
  const formatLabel = (key) => {
    return key
      .replace(/_/g, ' ')
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase())
      .trim();
  };

  // Generate PDF Report
  const generatePDFReport = async () => {
    try {
      setGeneratingPDF(true);
      
      // Extract base64 data without data URL prefix for PDF
      const getBase64Data = (base64Data) => {
        if (!base64Data) return null;
        if (base64Data.startsWith('data:')) {
          return base64Data.split(',')[1]; // Remove data URL prefix
        }
        return base64Data;
      };

      const reportData = {
        diameter,
        verdict,
        level1,
        level2,
        segmentedImageBase64: getBase64Data(segmented_image_base64),
        debugImageBase64: getBase64Data(debug_image_base64),
        testId,
        currentDate,
        currentTime
      };

      const uri = await pdfGenerator.generateAndSharePDF(reportData);
      setPdfUri(uri);
      
      Alert.alert(
        'Success!',
        'PDF report has been generated and shared successfully.',
        [{ text: 'OK' }]
      );
    } catch (error) {
      console.error('PDF generation error:', error);
      Alert.alert(
        'Error',
        'Failed to generate PDF report. Please try again.',
        [{ text: 'OK' }]
      );
    } finally {
      setGeneratingPDF(false);
    }
  };

  // Generate PDF for preview
  const generatePDFPreview = async () => {
    try {
      setGeneratingPDF(true);
      
      // Extract base64 data without data URL prefix for PDF
      const getBase64Data = (base64Data) => {
        if (!base64Data) return null;
        if (base64Data.startsWith('data:')) {
          return base64Data.split(',')[1]; // Remove data URL prefix
        }
        return base64Data;
      };

      const reportData = {
        diameter,
        verdict,
        level1,
        level2,
        segmentedImageBase64: getBase64Data(segmented_image_base64),
        debugImageBase64: getBase64Data(debug_image_base64),
        testId,
        currentDate,
        currentTime
      };

      const uri = await pdfGenerator.generatePDF(reportData);
      setPdfUri(uri);
      
      Alert.alert(
        'PDF Generated',
        'PDF report has been generated successfully. You can now share it.',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Share', onPress: () => sharePDF(uri) }
        ]
      );
    } catch (error) {
      console.error('PDF generation error:', error);
      Alert.alert(
        'Error',
        'Failed to generate PDF report. Please try again.',
        [{ text: 'OK' }]
      );
    } finally {
      setGeneratingPDF(false);
    }
  };

  // Share PDF
  const sharePDF = async (uri) => {
    try {
      const Sharing = require('expo-sharing');
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(uri, {
          mimeType: 'application/pdf',
          dialogTitle: 'TMT Ring Test Report'
        });
      }
    } catch (error) {
      console.error('Sharing error:', error);
      Alert.alert('Error', 'Failed to share PDF.');
    }
  };

  // Render Level 1 results
  const renderLevel1Results = () => {
    if (!level1) return null;

    const results = [
      { key: 'dark_grey_and_light_core_visible', label: 'Layers Detected' },
      { key: 'continuous_outer_ring', label: 'Continuous Outer Ring' },
      { key: 'concentric_regions', label: 'Concentric Regions' },
      { key: 'uniform_thickness', label: 'Uniform Thickness' }
    ];

    return (
      <View style={styles.resultsSection}>
        <View style={styles.sectionHeader}>
          <Icon name="color-lens" size={24} color="#00D4FF" />
          <Text style={styles.sectionTitle}>Level 1: Color & Shape Analysis</Text>
        </View>
        {results.map(({ key, label }) => (
          <View key={key} style={styles.resultRow}>
            <Text style={styles.resultLabel}>{label}</Text>
            <View style={styles.resultValueContainer}>
              <Text style={[
                styles.resultValue,
                level1[key] ? styles.passText : styles.failText
              ]}>
                {level1[key] ? 'Yes' : 'No'}
              </Text>
              <Icon
                name={level1[key] ? 'check-circle' : 'cancel'}
                size={20}
                color={level1[key] ? '#4CAF50' : '#f44336'}
              />
            </View>
          </View>
        ))}
      </View>
    );
  };

  // Render Level 2 results
  const renderLevel2Results = () => {
    if (!level2) return null;

    const minThickness = level2.min_thickness_mm || 0;
    const maxThickness = level2.max_thickness_mm || 0;
    const minPercentage = ((minThickness / diameter) * 100).toFixed(1);
    const maxPercentage = ((maxThickness / diameter) * 100).toFixed(1);

    const results = [
      { label: 'Minimum Thickness', value: `${minThickness} mm`, status: 'info' },
      { label: 'Maximum Thickness', value: `${maxThickness} mm`, status: 'info' },
      { label: 'Thickness Range', value: `${minThickness} - ${maxThickness} mm`, status: 'info' },
      { label: 'Thickness Percentage', value: `${minPercentage}% - ${maxPercentage}%`, status: 'info' },
      { label: 'Within Standard Range', value: level2.quality_status ? 'Yes' : 'No', status: level2.quality_status ? 'pass' : 'fail' },
      { label: 'Quality Status', value: level2.quality_status ? 'Meets Standards' : 'Below Standards', status: level2.quality_status ? 'pass' : 'fail' }
    ];

    return (
      <View style={styles.resultsSection}>
        <View style={styles.sectionHeader}>
          <Icon name="straighten" size={24} color="#00D4FF" />
          <Text style={styles.sectionTitle}>Level 2: Dimensional Analysis</Text>
        </View>
        {results.map(({ label, value, status }, index) => (
          <View key={index} style={styles.resultRow}>
            <Text style={styles.resultLabel}>{label}</Text>
            <View style={styles.resultValueContainer}>
              <Text style={[
                styles.resultValue,
                status === 'pass' ? styles.passText :
                status === 'fail' ? styles.failText : styles.infoText
              ]}>
                {value}
              </Text>
              {status !== 'info' && (
                <Icon
                  name={status === 'pass' ? 'check-circle' : 'cancel'}
                  size={20}
                  color={status === 'pass' ? '#4CAF50' : '#f44336'}
                />
              )}
            </View>
          </View>
        ))}
      </View>
    );
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <LinearGradient
        colors={['#0A0A0A', '#1A1A2E', '#16213E']}
        style={styles.header}
      >
        <View style={styles.logoContainer}>
          <View style={styles.logo}>
            <Text style={styles.logoText}>TATA</Text>
            <Text style={styles.logoSubText}>STEEL</Text>
          </View>
        </View>
        <Text style={styles.reportTitle}>TMT Bar Ring Test Report</Text>
        <Text style={styles.reportId}>Report ID: {testId}</Text>
      </LinearGradient>

      {/* Test Information */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Icon name="info" size={24} color="#00D4FF" />
          <Text style={styles.sectionTitle}>Test Information</Text>
        </View>
        <View style={styles.infoGrid}>
          <View style={styles.infoItem}>
            <Text style={styles.infoLabel}>Date</Text>
            <Text style={styles.infoValue}>{currentDate}</Text>
          </View>
          <View style={styles.infoItem}>
            <Text style={styles.infoLabel}>Time</Text>
            <Text style={styles.infoValue}>{currentTime}</Text>
          </View>
          <View style={styles.infoItem}>
            <Text style={styles.infoLabel}>Bar Diameter</Text>
            <Text style={styles.infoValue}>{diameter} mm</Text>
          </View>
          <View style={styles.infoItem}>
            <Text style={styles.infoLabel}>Test Type</Text>
            <Text style={styles.infoValue}>Ring Test (NITOL)</Text>
          </View>
        </View>
        
        <View style={[
          styles.statusBadge,
          verdict === 'PASS' ? styles.passBadge : styles.failBadge
        ]}>
          <Icon
            name={verdict === 'PASS' ? 'check-circle' : 'cancel'}
            size={30}
            color="white"
          />
          <Text style={styles.statusText}>Overall Result: {verdict}</Text>
        </View>
      </View>

      {/* Level 1 Results */}
      {renderLevel1Results()}

      {/* Level 1 Image */}
      {segmentedImageUri && (
        <View style={styles.imageSection}>
          <Text style={styles.sectionTitle}>Level 1: Segmented Image Analysis</Text>
          <Image
            source={{ uri: segmentedImageUri }}
            style={styles.analysisImage}
            resizeMode="contain"
          />
          <Text style={styles.imageCaption}>Cross-section analysis after NITOL application</Text>
        </View>
      )}

      {/* Level 2 Results */}
      {renderLevel2Results()}

      {/* Level 2 Image */}
      {debugImageUri && (
        <View style={styles.imageSection}>
          <Text style={styles.sectionTitle}>Level 2: Thickness Analysis</Text>
          <Image
            source={{ uri: debugImageUri }}
            style={styles.analysisImage}
            resizeMode="contain"
          />
          <Text style={styles.imageCaption}>Rim thickness measurement and analysis</Text>
        </View>
      )}

      {/* Quality Standards */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Icon name="rule" size={24} color="#00D4FF" />
          <Text style={styles.sectionTitle}>Quality Standards</Text>
        </View>
        <View style={styles.standardsList}>
          <View style={styles.standardItem}>
            <Icon name="check-circle-outline" size={16} color="#666" />
            <Text style={styles.standardText}>Rim thickness should be 7-10% of bar diameter</Text>
          </View>
          <View style={styles.standardItem}>
            <Icon name="check-circle-outline" size={16} color="#666" />
            <Text style={styles.standardText}>Three distinct layers: Rim (Dark), Transition, Core (Light)</Text>
          </View>
          <View style={styles.standardItem}>
            <Icon name="check-circle-outline" size={16} color="#666" />
            <Text style={styles.standardText}>Continuous outer ring structure</Text>
          </View>
          <View style={styles.standardItem}>
            <Icon name="check-circle-outline" size={16} color="#666" />
            <Text style={styles.standardText}>Concentric regions with uniform thickness</Text>
          </View>
        </View>
      </View>

      {/* Action Buttons */}
      <View style={styles.actionContainer}>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={generatePDFReport}
          disabled={generatingPDF}
        >
          <LinearGradient
            colors={['#00D4FF', '#0099CC']}
            style={styles.buttonGradient}
          >
            {generatingPDF ? (
              <ActivityIndicator size="small" color="white" />
            ) : (
              <Icon name="picture-as-pdf" size={24} color="white" />
            )}
            <Text style={styles.buttonText}>
              {generatingPDF ? 'Generating PDF...' : 'Generate & Share PDF'}
            </Text>
          </LinearGradient>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={generatePDFPreview}
          disabled={generatingPDF}
        >
          <Icon name="preview" size={24} color="#00D4FF" />
          <Text style={styles.secondaryButtonText}>Preview PDF</Text>
        </TouchableOpacity>

        {pdfUri && (
          <TouchableOpacity
            style={styles.tertiaryButton}
            onPress={() => sharePDF(pdfUri)}
          >
            <Icon name="share" size={24} color="#00D4FF" />
            <Text style={styles.tertiaryButtonText}>Share PDF</Text>
          </TouchableOpacity>
        )}

        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Icon name="arrow-back" size={24} color="#666" />
          <Text style={styles.backButtonText}>Back to Results</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A0A',
  },
  header: {
    padding: 30,
    alignItems: 'center',
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
  },
  logoContainer: {
    marginBottom: 20,
  },
  logo: {
    alignItems: 'center',
  },
  logoText: {
    fontSize: 32,
    fontWeight: '900',
    color: '#00D4FF',
    textShadowColor: 'rgba(0, 212, 255, 0.5)',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
  },
  logoSubText: {
    fontSize: 18,
    color: 'white',
    opacity: 0.9,
  },
  reportTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 10,
  },
  reportId: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  section: {
    backgroundColor: '#1A1A2E',
    margin: 20,
    marginTop: 10,
    padding: 25,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: '#00D4FF',
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    marginLeft: 10,
  },
  infoGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  infoItem: {
    width: '48%',
    backgroundColor: '#16213E',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#00D4FF',
  },
  infoLabel: {
    fontSize: 12,
    color: '#00D4FF',
    textTransform: 'uppercase',
    fontWeight: 'bold',
    marginBottom: 5,
  },
  infoValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 10,
    marginTop: 10,
  },
  passBadge: {
    backgroundColor: '#4CAF50',
  },
  failBadge: {
    backgroundColor: '#f44336',
  },
  statusText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    marginLeft: 10,
  },
  resultsSection: {
    backgroundColor: '#1A1A2E',
    margin: 20,
    marginTop: 10,
    padding: 25,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: '#00D4FF',
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#16213E',
  },
  resultLabel: {
    fontSize: 16,
    color: 'white',
    flex: 1,
  },
  resultValueContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  resultValue: {
    fontSize: 16,
    fontWeight: '600',
    marginRight: 8,
  },
  passText: {
    color: '#4CAF50',
  },
  failText: {
    color: '#f44336',
  },
  infoText: {
    color: '#00D4FF',
  },
  imageSection: {
    margin: 20,
    marginTop: 10,
  },
  analysisImage: {
    width: '100%',
    height: 200,
    borderRadius: 10,
    backgroundColor: '#16213E',
    borderWidth: 2,
    borderColor: '#00D4FF',
  },
  imageCaption: {
    fontSize: 14,
    color: '#666',
    marginTop: 10,
    fontStyle: 'italic',
    textAlign: 'center',
  },
  standardsList: {
    marginTop: 10,
  },
  standardItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#16213E',
  },
  standardText: {
    fontSize: 14,
    color: 'white',
    marginLeft: 10,
    flex: 1,
  },
  actionContainer: {
    padding: 20,
    paddingTop: 0,
  },
  primaryButton: {
    marginBottom: 15,
    borderRadius: 15,
    overflow: 'hidden',
    elevation: 5,
    shadowColor: '#00D4FF',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
  },
  buttonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 18,
    paddingHorizontal: 30,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 10,
  },
  secondaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1A1A2E',
    paddingVertical: 18,
    paddingHorizontal: 30,
    borderRadius: 15,
    borderWidth: 2,
    borderColor: '#00D4FF',
    marginBottom: 15,
  },
  secondaryButtonText: {
    color: '#00D4FF',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 10,
  },
  tertiaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#16213E',
    paddingVertical: 18,
    paddingHorizontal: 30,
    borderRadius: 15,
    borderWidth: 2,
    borderColor: '#00D4FF',
    marginBottom: 15,
  },
  tertiaryButtonText: {
    color: '#00D4FF',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 10,
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1A1A2E',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: '#666',
  },
  backButtonText: {
    color: '#666',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 10,
  },
});

export default ReportScreen;