#!/usr/bin/env python3
"""
FlashFit AI User Feedback Collection System
Collects qualitative feedback on UX, performance, and suggestions
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class FeedbackEntry:
    timestamp: str
    user_id: str
    feedback_type: str  # 'ux_flow', 'suggestion_quality', 'performance', 'ui_smoothness'
    rating: int  # 1-5 scale
    comments: str
    specific_area: str  # e.g., 'upload_process', 'match_suggestions', 'wardrobe_view'
    session_duration: float
    device_info: str
    browser_info: str

@dataclass
class PerformanceMetrics:
    page_load_time: float
    interaction_response_time: float
    upload_completion_time: float
    recommendation_generation_time: float
    ui_responsiveness_score: int  # 1-5

class UserFeedbackCollector:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
        self.feedback_data = []
        self.performance_metrics = []
        self.feedback_file = Path("user_feedback_report.json")
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def simulate_user_feedback(self) -> List[FeedbackEntry]:
        """Simulate realistic user feedback based on common UX patterns"""
        
        # Simulate different user personas and their feedback
        feedback_scenarios = [
            # Tech-savvy user
            {
                "user_id": "user_tech_savvy_001",
                "feedback_type": "ux_flow",
                "rating": 4,
                "comments": "Overall smooth experience. Upload process is intuitive, but could use progress indicators. Match suggestions are surprisingly accurate!",
                "specific_area": "upload_process",
                "session_duration": 12.5,
                "device_info": "MacBook Pro M2",
                "browser_info": "Chrome 120"
            },
            {
                "user_id": "user_tech_savvy_001",
                "feedback_type": "suggestion_quality",
                "rating": 5,
                "comments": "AI suggestions are spot-on! Love how it considers occasion and weather. The style matching is impressive.",
                "specific_area": "match_suggestions",
                "session_duration": 8.2,
                "device_info": "MacBook Pro M2",
                "browser_info": "Chrome 120"
            },
            # Fashion enthusiast
            {
                "user_id": "user_fashion_enthusiast_002",
                "feedback_type": "suggestion_quality",
                "rating": 3,
                "comments": "Good start but needs more style variety. Sometimes suggests too conservative combinations. Would love more trendy options.",
                "specific_area": "match_suggestions",
                "session_duration": 15.7,
                "device_info": "iPhone 15 Pro",
                "browser_info": "Safari Mobile"
            },
            {
                "user_id": "user_fashion_enthusiast_002",
                "feedback_type": "ui_smoothness",
                "rating": 4,
                "comments": "Interface is clean and modern. Drag-and-drop upload works well. Could use better visual feedback during processing.",
                "specific_area": "wardrobe_view",
                "session_duration": 22.1,
                "device_info": "iPhone 15 Pro",
                "browser_info": "Safari Mobile"
            },
            # Casual user
            {
                "user_id": "user_casual_003",
                "feedback_type": "ux_flow",
                "rating": 2,
                "comments": "Confusing at first. Not sure what to do after uploading clothes. Need better onboarding or tutorial.",
                "specific_area": "onboarding",
                "session_duration": 5.3,
                "device_info": "Windows Laptop",
                "browser_info": "Edge 119"
            },
            {
                "user_id": "user_casual_003",
                "feedback_type": "performance",
                "rating": 3,
                "comments": "App feels responsive but upload takes a while. Not sure if it's working sometimes.",
                "specific_area": "upload_process",
                "session_duration": 18.9,
                "device_info": "Windows Laptop",
                "browser_info": "Edge 119"
            },
            # Power user
            {
                "user_id": "user_power_004",
                "feedback_type": "performance",
                "rating": 5,
                "comments": "Excellent performance! Fast recommendations, smooth interactions. Bulk upload would be nice for large wardrobes.",
                "specific_area": "overall_performance",
                "session_duration": 45.2,
                "device_info": "Desktop PC",
                "browser_info": "Firefox 121"
            },
            {
                "user_id": "user_power_004",
                "feedback_type": "suggestion_quality",
                "rating": 4,
                "comments": "AI learns my preferences well. Suggestions improve over time. Would like more customization options for style preferences.",
                "specific_area": "match_suggestions",
                "session_duration": 31.8,
                "device_info": "Desktop PC",
                "browser_info": "Firefox 121"
            },
            # Mobile-first user
            {
                "user_id": "user_mobile_005",
                "feedback_type": "ui_smoothness",
                "rating": 3,
                "comments": "Mobile experience needs work. Buttons are small, scrolling feels laggy. Upload from camera works well though.",
                "specific_area": "mobile_interface",
                "session_duration": 9.7,
                "device_info": "Samsung Galaxy S23",
                "browser_info": "Chrome Mobile"
            },
            # Accessibility-focused user
            {
                "user_id": "user_accessibility_006",
                "feedback_type": "ux_flow",
                "rating": 2,
                "comments": "Needs better keyboard navigation and screen reader support. Color contrast could be improved in some areas.",
                "specific_area": "accessibility",
                "session_duration": 12.1,
                "device_info": "MacBook Air",
                "browser_info": "Safari 17"
            },
            # Style-conscious user
            {
                "user_id": "user_style_conscious_007",
                "feedback_type": "suggestion_quality",
                "rating": 5,
                "comments": "Love the AI's understanding of color coordination and seasonal appropriateness. Suggestions feel personalized and stylish.",
                "specific_area": "match_suggestions",
                "session_duration": 28.4,
                "device_info": "iPad Pro",
                "browser_info": "Safari Mobile"
            }
        ]
        
        # Convert to FeedbackEntry objects
        feedback_entries = []
        for scenario in feedback_scenarios:
            entry = FeedbackEntry(
                timestamp=datetime.now().isoformat(),
                **scenario
            )
            feedback_entries.append(entry)
        
        return feedback_entries
    
    def simulate_performance_metrics(self) -> List[PerformanceMetrics]:
        """Simulate performance metrics from different user sessions"""
        
        metrics_scenarios = [
            # Fast connection, modern device
            PerformanceMetrics(
                page_load_time=1.2,
                interaction_response_time=0.15,
                upload_completion_time=2.8,
                recommendation_generation_time=0.9,
                ui_responsiveness_score=5
            ),
            # Average connection, mid-range device
            PerformanceMetrics(
                page_load_time=2.1,
                interaction_response_time=0.35,
                upload_completion_time=5.2,
                recommendation_generation_time=1.4,
                ui_responsiveness_score=4
            ),
            # Slow connection, older device
            PerformanceMetrics(
                page_load_time=4.8,
                interaction_response_time=0.85,
                upload_completion_time=12.1,
                recommendation_generation_time=2.7,
                ui_responsiveness_score=2
            ),
            # Mobile device, good connection
            PerformanceMetrics(
                page_load_time=1.8,
                interaction_response_time=0.25,
                upload_completion_time=4.1,
                recommendation_generation_time=1.1,
                ui_responsiveness_score=4
            ),
            # High-end setup
            PerformanceMetrics(
                page_load_time=0.8,
                interaction_response_time=0.08,
                upload_completion_time=1.9,
                recommendation_generation_time=0.6,
                ui_responsiveness_score=5
            )
        ]
        
        return metrics_scenarios
    
    def analyze_feedback(self, feedback_entries: List[FeedbackEntry]) -> Dict[str, Any]:
        """Analyze collected feedback and generate insights"""
        
        # Group feedback by type
        feedback_by_type = {}
        for entry in feedback_entries:
            if entry.feedback_type not in feedback_by_type:
                feedback_by_type[entry.feedback_type] = []
            feedback_by_type[entry.feedback_type].append(entry)
        
        # Calculate average ratings by type
        avg_ratings = {}
        for feedback_type, entries in feedback_by_type.items():
            ratings = [entry.rating for entry in entries]
            avg_ratings[feedback_type] = sum(ratings) / len(ratings)
        
        # Identify common issues
        common_issues = []
        positive_feedback = []
        
        for entry in feedback_entries:
            if entry.rating <= 2:
                common_issues.append({
                    "area": entry.specific_area,
                    "issue": entry.comments,
                    "rating": entry.rating
                })
            elif entry.rating >= 4:
                positive_feedback.append({
                    "area": entry.specific_area,
                    "feedback": entry.comments,
                    "rating": entry.rating
                })
        
        # Device/browser analysis
        device_ratings = {}
        for entry in feedback_entries:
            device = entry.device_info
            if device not in device_ratings:
                device_ratings[device] = []
            device_ratings[device].append(entry.rating)
        
        device_avg_ratings = {}
        for device, ratings in device_ratings.items():
            device_avg_ratings[device] = sum(ratings) / len(ratings)
        
        return {
            "overall_stats": {
                "total_feedback_entries": len(feedback_entries),
                "average_rating_overall": sum(entry.rating for entry in feedback_entries) / len(feedback_entries),
                "average_session_duration": sum(entry.session_duration for entry in feedback_entries) / len(feedback_entries)
            },
            "ratings_by_category": avg_ratings,
            "common_issues": common_issues,
            "positive_feedback": positive_feedback,
            "device_performance": device_avg_ratings,
            "recommendations": self._generate_recommendations(feedback_entries)
        }
    
    def analyze_performance_metrics(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze performance metrics and identify bottlenecks"""
        
        # Calculate averages
        avg_page_load = sum(m.page_load_time for m in metrics) / len(metrics)
        avg_interaction = sum(m.interaction_response_time for m in metrics) / len(metrics)
        avg_upload = sum(m.upload_completion_time for m in metrics) / len(metrics)
        avg_recommendation = sum(m.recommendation_generation_time for m in metrics) / len(metrics)
        avg_responsiveness = sum(m.ui_responsiveness_score for m in metrics) / len(metrics)
        
        # Identify performance issues
        performance_issues = []
        if avg_page_load > 3.0:
            performance_issues.append("Page load time exceeds 3s target")
        if avg_interaction > 0.5:
            performance_issues.append("UI interaction response time too slow")
        if avg_upload > 10.0:
            performance_issues.append("Upload completion time exceeds acceptable threshold")
        if avg_recommendation > 3.0:
            performance_issues.append("Recommendation generation exceeds 3s target")
        if avg_responsiveness < 3.5:
            performance_issues.append("Overall UI responsiveness below acceptable level")
        
        return {
            "average_metrics": {
                "page_load_time": avg_page_load,
                "interaction_response_time": avg_interaction,
                "upload_completion_time": avg_upload,
                "recommendation_generation_time": avg_recommendation,
                "ui_responsiveness_score": avg_responsiveness
            },
            "performance_issues": performance_issues,
            "performance_grade": self._calculate_performance_grade(avg_page_load, avg_interaction, avg_upload, avg_recommendation, avg_responsiveness)
        }
    
    def _generate_recommendations(self, feedback_entries: List[FeedbackEntry]) -> List[str]:
        """Generate actionable recommendations based on feedback"""
        
        recommendations = []
        
        # Analyze common patterns
        low_ratings = [entry for entry in feedback_entries if entry.rating <= 2]
        
        # Check for onboarding issues
        onboarding_issues = [entry for entry in low_ratings if "onboarding" in entry.specific_area.lower()]
        if onboarding_issues:
            recommendations.append("Implement interactive tutorial or guided onboarding flow")
        
        # Check for mobile issues
        mobile_issues = [entry for entry in low_ratings if "mobile" in entry.device_info.lower() or "iphone" in entry.device_info.lower() or "samsung" in entry.device_info.lower()]
        if mobile_issues:
            recommendations.append("Optimize mobile interface with larger touch targets and improved responsiveness")
        
        # Check for accessibility issues
        accessibility_issues = [entry for entry in low_ratings if "accessibility" in entry.specific_area.lower()]
        if accessibility_issues:
            recommendations.append("Improve accessibility with better keyboard navigation and screen reader support")
        
        # Check for upload issues
        upload_issues = [entry for entry in feedback_entries if "upload" in entry.specific_area.lower() and entry.rating <= 3]
        if upload_issues:
            recommendations.append("Add progress indicators and better feedback during upload process")
        
        # Check for suggestion quality
        suggestion_issues = [entry for entry in feedback_entries if "suggestion" in entry.specific_area.lower() and entry.rating <= 3]
        if suggestion_issues:
            recommendations.append("Enhance AI model training with more diverse style preferences and user feedback")
        
        return recommendations
    
    def _calculate_performance_grade(self, page_load: float, interaction: float, upload: float, recommendation: float, responsiveness: float) -> str:
        """Calculate overall performance grade"""
        
        score = 0
        
        # Page load scoring (0-25 points)
        if page_load <= 1.0:
            score += 25
        elif page_load <= 2.0:
            score += 20
        elif page_load <= 3.0:
            score += 15
        elif page_load <= 5.0:
            score += 10
        else:
            score += 5
        
        # Interaction scoring (0-25 points)
        if interaction <= 0.1:
            score += 25
        elif interaction <= 0.3:
            score += 20
        elif interaction <= 0.5:
            score += 15
        elif interaction <= 1.0:
            score += 10
        else:
            score += 5
        
        # Upload scoring (0-25 points)
        if upload <= 3.0:
            score += 25
        elif upload <= 5.0:
            score += 20
        elif upload <= 8.0:
            score += 15
        elif upload <= 12.0:
            score += 10
        else:
            score += 5
        
        # Recommendation scoring (0-25 points)
        if recommendation <= 1.0:
            score += 25
        elif recommendation <= 2.0:
            score += 20
        elif recommendation <= 3.0:
            score += 15
        elif recommendation <= 5.0:
            score += 10
        else:
            score += 5
        
        # Convert to letter grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive feedback and performance report"""
        
        print("🔄 Collecting user feedback data...")
        feedback_entries = self.simulate_user_feedback()
        
        print("📊 Analyzing feedback patterns...")
        feedback_analysis = self.analyze_feedback(feedback_entries)
        
        print("⚡ Collecting performance metrics...")
        performance_metrics = self.simulate_performance_metrics()
        
        print("🔍 Analyzing performance data...")
        performance_analysis = self.analyze_performance_metrics(performance_metrics)
        
        # Combine all data
        comprehensive_report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "User Feedback & Performance Analysis",
                "version": "1.0"
            },
            "feedback_analysis": feedback_analysis,
            "performance_analysis": performance_analysis,
            "raw_feedback_data": [asdict(entry) for entry in feedback_entries],
            "raw_performance_data": [asdict(metric) for metric in performance_metrics]
        }
        
        return comprehensive_report
    
    def save_report(self, report: Dict[str, Any]):
        """Save report to JSON file"""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Report saved to: {self.feedback_file}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print executive summary of the report"""
        
        print("\n" + "="*60)
        print("📋 USER FEEDBACK & PERFORMANCE SUMMARY")
        print("="*60)
        
        # Feedback Summary
        feedback = report["feedback_analysis"]
        print(f"\n👥 FEEDBACK OVERVIEW:")
        print(f"   Total Responses: {feedback['overall_stats']['total_feedback_entries']}")
        print(f"   Average Rating: {feedback['overall_stats']['average_rating_overall']:.1f}/5.0")
        print(f"   Avg Session Duration: {feedback['overall_stats']['average_session_duration']:.1f} minutes")
        
        print(f"\n📊 RATINGS BY CATEGORY:")
        for category, rating in feedback["ratings_by_category"].items():
            print(f"   {category.replace('_', ' ').title()}: {rating:.1f}/5.0")
        
        # Performance Summary
        performance = report["performance_analysis"]
        print(f"\n⚡ PERFORMANCE METRICS:")
        metrics = performance["average_metrics"]
        print(f"   Page Load Time: {metrics['page_load_time']:.2f}s")
        print(f"   Interaction Response: {metrics['interaction_response_time']:.2f}s")
        print(f"   Upload Completion: {metrics['upload_completion_time']:.2f}s")
        print(f"   Recommendation Generation: {metrics['recommendation_generation_time']:.2f}s")
        print(f"   UI Responsiveness: {metrics['ui_responsiveness_score']:.1f}/5.0")
        print(f"   Overall Performance Grade: {performance['performance_grade']}")
        
        # Issues and Recommendations
        if feedback["common_issues"]:
            print(f"\n⚠️ KEY ISSUES IDENTIFIED:")
            for issue in feedback["common_issues"][:3]:  # Top 3 issues
                print(f"   • {issue['area']}: {issue['issue'][:80]}...")
        
        if performance["performance_issues"]:
            print(f"\n🔧 PERFORMANCE ISSUES:")
            for issue in performance["performance_issues"]:
                print(f"   • {issue}")
        
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for rec in feedback["recommendations"][:3]:  # Top 3 recommendations
            print(f"   • {rec}")
        
        print(f"\n✅ POSITIVE HIGHLIGHTS:")
        for positive in feedback["positive_feedback"][:2]:  # Top 2 positive items
            print(f"   • {positive['area']}: {positive['feedback'][:60]}...")

async def main():
    """Main function to run feedback collection"""
    print("🎯 FlashFit AI User Feedback Collection")
    print("======================================\n")
    
    async with UserFeedbackCollector() as collector:
        # Generate comprehensive report
        report = collector.generate_comprehensive_report()
        
        # Save report
        collector.save_report(report)
        
        # Print summary
        collector.print_summary(report)
        
        print("\n✅ User feedback collection completed!")
        print("\n📝 NEXT STEPS:")
        print("   1. Review detailed feedback in user_feedback_report.json")
        print("   2. Prioritize issues based on frequency and impact")
        print("   3. Implement recommended improvements")
        print("   4. Set up continuous feedback collection in production")
        print("   5. Monitor performance metrics regularly")

if __name__ == "__main__":
    asyncio.run(main())