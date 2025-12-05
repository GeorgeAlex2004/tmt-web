import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Dimensions,
  Animated,
  FlatList,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import Icon from 'react-native-vector-icons/MaterialIcons';

const { width } = Dimensions.get('window');

const HistoryScreen = () => {
  const navigation = useNavigation();
  const [selectedFilter, setSelectedFilter] = useState('all');
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;
  const glowAnim = useRef(new Animated.Value(0)).current;

  // Mock data for history
  const mockHistoryData = [
    {
      id: '1',
      testType: 'Ring Test',
      diameter: '16mm',
      date: '2024-01-15',
      time: '14:30',
      status: 'Passed',
      accuracy: '98.5%',
      imageUri: null,
    },
    {
      id: '2',
      testType: 'Rib Test',
      diameter: '20mm',
      date: '2024-01-14',
      time: '16:45',
      status: 'Passed',
      accuracy: '97.2%',
      imageUri: null,
    },
    {
      id: '3',
      testType: 'Ring Test',
      diameter: '12mm',
      date: '2024-01-13',
      time: '09:15',
      status: 'Failed',
      accuracy: '85.1%',
      imageUri: null,
    },
    {
      id: '4',
      testType: 'Rib Test',
      diameter: '25mm',
      date: '2024-01-12',
      time: '11:20',
      status: 'Passed',
      accuracy: '99.1%',
      imageUri: null,
    },
    {
      id: '5',
      testType: 'Ring Test',
      diameter: '16mm',
      date: '2024-01-11',
      time: '13:45',
      status: 'Passed',
      accuracy: '96.8%',
      imageUri: null,
    },
  ];

  const filters = [
    { key: 'all', label: 'All Tests', icon: 'list' },
    { key: 'ring', label: 'Ring Tests', icon: 'science' },
    { key: 'rib', label: 'Rib Tests', icon: 'straighten' },
    { key: 'passed', label: 'Passed', icon: 'check-circle' },
    { key: 'failed', label: 'Failed', icon: 'error' },
  ];

  useEffect(() => {
    // Start entrance animations
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 600,
        useNativeDriver: true,
      }),
    ]).start();

    // Start glow animation
    Animated.loop(
      Animated.sequence([
        Animated.timing(glowAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }),
        Animated.timing(glowAnim, {
          toValue: 0,
          duration: 2000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);

  const handleGoBack = () => {
    navigation.goBack();
  };

  const handleFilterSelect = (filterKey) => {
    setSelectedFilter(filterKey);
  };

  const handleViewDetails = (testData) => {
    // Navigate to test details screen
    navigation.navigate('TestDetails', { testData });
  };

  const getFilteredData = () => {
    switch (selectedFilter) {
      case 'ring':
        return mockHistoryData.filter(item => item.testType === 'Ring Test');
      case 'rib':
        return mockHistoryData.filter(item => item.testType === 'Rib Test');
      case 'passed':
        return mockHistoryData.filter(item => item.status === 'Passed');
      case 'failed':
        return mockHistoryData.filter(item => item.status === 'Failed');
      default:
        return mockHistoryData;
    }
  };

  const renderHistoryItem = ({ item, index }) => {
    const isPassed = item.status === 'Passed';
    
    return (
      <Animated.View
        style={[
          styles.historyItem,
          {
            opacity: fadeAnim,
            transform: [
              {
                translateY: fadeAnim.interpolate({
                  inputRange: [0, 1],
                  outputRange: [30, 0],
                }),
              },
            ],
          },
        ]}
      >
        <LinearGradient
          colors={isPassed ? ['rgba(76, 175, 80, 0.1)', 'rgba(76, 175, 80, 0.05)'] : ['rgba(244, 67, 54, 0.1)', 'rgba(244, 67, 54, 0.05)']}
          style={styles.itemGradient}
        >
          <View style={styles.itemHeader}>
            <View style={styles.itemLeft}>
              <Icon 
                name={item.testType === 'Ring Test' ? 'science' : 'straighten'} 
                size={24} 
                color={isPassed ? '#4CAF50' : '#F44336'} 
              />
              <View style={styles.itemInfo}>
                <Text style={styles.itemTitle}>{item.testType}</Text>
                <Text style={styles.itemSubtitle}>{item.diameter} • {item.date} at {item.time}</Text>
              </View>
            </View>
            <View style={styles.itemRight}>
              <View style={[styles.statusBadge, { backgroundColor: isPassed ? 'rgba(76, 175, 80, 0.2)' : 'rgba(244, 67, 54, 0.2)' }]}>
                <Text style={[styles.statusText, { color: isPassed ? '#4CAF50' : '#F44336' }]}>
                  {item.status}
                </Text>
              </View>
            </View>
          </View>
          
          <View style={styles.itemDetails}>
            <View style={styles.detailItem}>
              <Icon name="analytics" size={16} color="#00D4FF" />
              <Text style={styles.detailText}>Accuracy: {item.accuracy}</Text>
            </View>
            <TouchableOpacity style={styles.viewButton} onPress={() => handleViewDetails(item)}>
              <Text style={styles.viewButtonText}>View Details</Text>
              <Icon name="arrow-forward" size={16} color="#00D4FF" />
            </TouchableOpacity>
          </View>
        </LinearGradient>
      </Animated.View>
    );
  };

  const glowOpacity = glowAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0.3, 1],
  });

  return (
    <LinearGradient
      colors={['#0A0A0A', '#1A1A2E', '#16213E', '#0F3460']}
      style={styles.container}
    >
      {/* Animated Background Elements */}
      <Animated.View style={[styles.backgroundCircle, { opacity: glowOpacity }]} />
      <Animated.View style={[styles.backgroundCircle2, { opacity: glowOpacity }]} />
      
      <ScrollView 
        style={styles.scrollContainer}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header Section */}
        <Animated.View
          style={[
            styles.headerContent,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            },
          ]}
        >
          {/* Back Button */}
          <TouchableOpacity
            style={styles.backButton}
            onPress={handleGoBack}
          >
            <Icon name="arrow-back" size={24} color="#00D4FF" />
          </TouchableOpacity>

          <View style={styles.logoContainer}>
            <Icon name="analytics" size={50} color="#00D4FF" />
          </View>
          <Text style={styles.headerTitle}>Test History</Text>
          <Text style={styles.headerSubtitle}>
            View your analysis results and analytics
          </Text>
        </Animated.View>

        {/* Filter Section */}
        <Animated.View 
          style={[
            styles.filterSection,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            },
          ]}
        >
          <Text style={styles.sectionTitle}>Filter Results</Text>
          <ScrollView 
            horizontal 
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.filterContainer}
          >
            {filters.map((filter) => (
              <TouchableOpacity
                key={filter.key}
                style={[
                  styles.filterButton,
                  selectedFilter === filter.key && styles.selectedFilter,
                ]}
                onPress={() => handleFilterSelect(filter.key)}
              >
                <Icon 
                  name={filter.icon} 
                  size={20} 
                  color={selectedFilter === filter.key ? '#FFFFFF' : '#00D4FF'} 
                />
                <Text style={[
                  styles.filterText,
                  selectedFilter === filter.key && styles.selectedFilterText,
                ]}>
                  {filter.label}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </Animated.View>

        {/* Statistics Section */}
        <Animated.View 
          style={[
            styles.statsSection,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            },
          ]}
        >
          <Text style={styles.sectionTitle}>Summary</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statCard}>
              <Icon name="check-circle" size={24} color="#4CAF50" />
              <Text style={styles.statNumber}>4</Text>
              <Text style={styles.statLabel}>Passed Tests</Text>
            </View>
            <View style={styles.statCard}>
              <Icon name="error" size={24} color="#F44336" />
              <Text style={styles.statNumber}>1</Text>
              <Text style={styles.statLabel}>Failed Tests</Text>
            </View>
            <View style={styles.statCard}>
              <Icon name="analytics" size={24} color="#00D4FF" />
              <Text style={styles.statNumber}>96.8%</Text>
              <Text style={styles.statLabel}>Avg Accuracy</Text>
            </View>
          </View>
        </Animated.View>

        {/* History List */}
        <Animated.View 
          style={[
            styles.historySection,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            },
          ]}
        >
          <Text style={styles.sectionTitle}>Recent Tests</Text>
          <FlatList
            data={getFilteredData()}
            renderItem={renderHistoryItem}
            keyExtractor={(item) => item.id}
            scrollEnabled={false}
            showsVerticalScrollIndicator={false}
          />
        </Animated.View>
      </ScrollView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  backgroundCircle: {
    position: 'absolute',
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    top: 20,
    right: -50,
  },
  backgroundCircle2: {
    position: 'absolute',
    width: 150,
    height: 150,
    borderRadius: 75,
    backgroundColor: 'rgba(0, 212, 255, 0.05)',
    bottom: 20,
    left: -30,
  },
  scrollContainer: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingTop: 60,
  },
  headerContent: {
    alignItems: 'center',
    paddingTop: 20,
    paddingBottom: 30,
  },
  backButton: {
    position: 'absolute',
    top: 20,
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
  logoContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    borderWidth: 2,
    borderColor: 'rgba(0, 212, 255, 0.3)',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    lineHeight: 22,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 15,
  },
  filterSection: {
    marginBottom: 30,
  },
  filterContainer: {
    paddingHorizontal: 5,
  },
  filterButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    marginRight: 12,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  selectedFilter: {
    backgroundColor: 'rgba(0, 212, 255, 0.2)',
    borderColor: '#00D4FF',
  },
  filterText: {
    fontSize: 14,
    color: '#00D4FF',
    marginLeft: 8,
    fontWeight: '500',
  },
  selectedFilterText: {
    color: '#FFFFFF',
  },
  statsSection: {
    marginBottom: 30,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statCard: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    marginHorizontal: 5,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginTop: 8,
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
  },
  historySection: {
    marginBottom: 30,
  },
  historyItem: {
    marginBottom: 15,
    borderRadius: 16,
    overflow: 'hidden',
  },
  itemGradient: {
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  itemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  itemInfo: {
    marginLeft: 12,
    flex: 1,
  },
  itemTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 4,
  },
  itemSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.7)',
  },
  itemRight: {
    alignItems: 'flex-end',
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  statusText: {
    fontSize: 12,
    fontWeight: 'bold',
  },
  itemDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  detailItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  detailText: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    marginLeft: 8,
  },
  viewButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.3)',
  },
  viewButtonText: {
    fontSize: 12,
    color: '#00D4FF',
    fontWeight: '500',
    marginRight: 4,
  },
});

export default HistoryScreen; 