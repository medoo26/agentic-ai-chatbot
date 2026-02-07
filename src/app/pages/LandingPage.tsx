import { useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { GraduationCap, MessageSquare } from 'lucide-react';

export function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-[#E8F5E9] flex flex-col items-center justify-center p-6">
      <div className="max-w-2xl w-full text-center space-y-8">
        
        {/* Logo */}
        <div className="flex justify-center mb-6">
          <div className="w-32 h-32 bg-[#2E7D32] rounded-full flex items-center justify-center shadow-lg">
            <GraduationCap className="w-16 h-16 text-white" />
          </div>
        </div>

        {/* Title */}
        <div className="space-y-4">
          <h1 className="text-4xl font-bold text-[#2E7D32]">
            المساعد  الأكاديمي – جامعة الأمير سطّام
          </h1>
          <p className="text-lg text-gray-600">
            مساعد ذكي للإجابة على الاستفسارات الأكاديمية والإدارية
          </p>
        </div>

        {/* CTA Button */}
        <div className="pt-6">
          <Button
            onClick={() => navigate('/chat')}
            className="bg-[#2E7D32] hover:bg-[#1B5E20] text-white px-12 py-6 text-xl rounded-xl shadow-lg transition-all hover:shadow-xl"
          >
            <MessageSquare className="ml-3 h-6 w-6" />
            ابدأ المحادثة
          </Button>
        </div>

        {/* Admin Link */}
        <div className="pt-8">
          <button
            onClick={() => navigate('/admin/login')}
            className="text-[#2E7D32] hover:underline text-sm"
          >
            دخول الإدارة
          </button>
        </div>

      </div>
    </div>
  );
}
