// User Types
export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

// Admin Types
export interface Document {
  id: string;
  name: string;
  category: 'أكاديمي' | 'إداري';
  uploadDate: Date;
  status: 'نشط' | 'غير نشط';
  size: string;
}

export interface KnowledgeEntry {
  id: string;
  title: string;
  content: string;
  category: 'أكاديمي' | 'إداري';
  enabled: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface ChatLog {
  id: string;
  userType: string; // بدون رول (نص عام)
  question: string;
  answer: string;
  timestamp: Date;
  // ❌ rating انشال لأنك شلت التقييم
}

export interface SystemSettings {
  chatbotAvailable: boolean;
  systemMessage: string;
  modelStatus: 'نشط' | 'غير نشط';
  lastUpdate: Date;
}

export interface DashboardStats {
  totalDocuments: number;
  totalChats: number;
  activeUsers: number;
  avgResponseTime: string;
}
