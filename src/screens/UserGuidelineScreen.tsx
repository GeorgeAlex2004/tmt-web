import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import Icon from 'react-native-vector-icons/MaterialIcons';

const UserGuidelineScreen: React.FC = () => {
  const navigation = useNavigation<any>();

  const handleProceed = () => {
    navigation.navigate('Camera');
  };

  return (
    <LinearGradient
      colors={['#0A0A0A', '#1A1A2E', '#16213E', '#0F3460']}
      style={styles.container}
    >
      <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
        <Icon name="arrow-back" size={28} color="#00D4FF" />
      </TouchableOpacity>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.headerContent}>
          <Icon name="photo-camera" size={60} color="#00D4FF" style={styles.headerIcon} />
          <Text style={styles.title}>Photo Guidelines</Text>
          <Text style={styles.subtitle}>Follow these steps for best results</Text>
        </View>
        <View style={styles.guidelinesGrid}>
          <View style={styles.guidelineItem}>
            <Icon name="lightbulb-outline" size={36} color="#00D4FF" />
            <Text style={styles.guidelineTitle}>Good Lighting</Text>
            <Text style={styles.guidelineDescription}>Ensure the bar is well-lit and avoid shadows. Natural light is best.</Text>
          </View>
          <View style={styles.guidelineItem}>
            <Icon name="straighten" size={36} color="#00D4FF" />
            <Text style={styles.guidelineTitle}>Align the Bar</Text>
            <Text style={styles.guidelineDescription}>Hold the bar straight and parallel to the camera.</Text>
          </View>
          <View style={styles.guidelineItem}>
            <Icon name="blur-off" size={36} color="#00D4FF" />
            <Text style={styles.guidelineTitle}>Avoid Blur</Text>
            <Text style={styles.guidelineDescription}>Keep the camera steady and avoid shaking.</Text>
          </View>
          <View style={styles.guidelineItem}>
            <Icon name="crop-free" size={36} color="#00D4FF" />
            <Text style={styles.guidelineTitle}>Show Full Rib</Text>
            <Text style={styles.guidelineDescription}>Ensure the full rib is visible in the frame.</Text>
          </View>
          <View style={styles.guidelineItem}>
            <Icon name="filter-none" size={36} color="#00D4FF" />
            <Text style={styles.guidelineTitle}>Simple Background</Text>
            <Text style={styles.guidelineDescription}>Use a plain background to avoid distractions.</Text>
          </View>
          <View style={styles.guidelineItem}>
            <Icon name="view-module" size={36} color="#00D4FF" />
            <Text style={styles.guidelineTitle}>Show 10+ Ribs</Text>
            <Text style={styles.guidelineDescription}>At least 10 ribs should be clearly visible in the photo for accurate analysis.</Text>
          </View>
          <View style={styles.guidelineItem}>
            <Icon name="report-problem" size={36} color="#00D4FF" />
            <Text style={styles.guidelineTitle}>No Defects</Text>
            <Text style={styles.guidelineDescription}>Ensure none of the ribs have visible defects, rust, or damage.</Text>
          </View>
          <View style={styles.guidelineItem}>
            <Icon name="stay-primary-portrait" size={36} color="#00D4FF" />
            <Text style={styles.guidelineTitle}>Hold Steady</Text>
            <Text style={styles.guidelineDescription}>Hold your device steady and avoid movement.</Text>
          </View>
        </View>
        <TouchableOpacity style={styles.proceedButton} onPress={handleProceed} activeOpacity={0.85}>
          <LinearGradient colors={['#00D4FF', '#0099CC']} style={styles.proceedButtonGradient}>
            <Text style={styles.proceedButtonText}>Proceed to Camera</Text>
          </LinearGradient>
        </TouchableOpacity>
      </ScrollView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollContent: {
    padding: 24,
    paddingTop: 60,
    paddingBottom: 40,
  },
  headerContent: {
    alignItems: 'center',
    marginBottom: 32,
  },
  headerIcon: {
    marginBottom: 12,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 8,
    letterSpacing: 1,
  },
  subtitle: {
    fontSize: 16,
    color: '#00D4FF',
    marginBottom: 20,
    fontWeight: '600',
    textAlign: 'center',
  },
  guidelinesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 32,
  },
  guidelineItem: {
    width: '48%',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 16,
    padding: 18,
    marginBottom: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  guidelineTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 4,
  },
  guidelineDescription: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
    lineHeight: 16,
  },
  proceedButton: {
    borderRadius: 25,
    overflow: 'hidden',
    marginTop: 10,
    elevation: 6,
    shadowColor: '#00D4FF',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  proceedButtonGradient: {
    paddingVertical: 16,
    alignItems: 'center',
  },
  proceedButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
  },
  backButton: {
    position: 'absolute',
    top: 30,
    left: 20,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.3)',
    zIndex: 10,
  },
});

export default UserGuidelineScreen; 