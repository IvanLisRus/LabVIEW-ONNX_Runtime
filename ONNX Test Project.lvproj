<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="17008000">
	<Property Name="NI.LV.All.SaveVersion" Type="Str">17.0</Property>
	<Property Name="NI.LV.All.SourceOnly" Type="Bool">true</Property>
	<Property Name="NI.Project.Description" Type="Str">Created IvanLis by LabVIEW Portal
Profile: http://labviewportal.org/memberlist.php?mode=viewprofile&amp;u=987
eMail: IvanLisanov@gMail.com
Telegram: https://t.me/IvanLis</Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="data" Type="Folder">
			<Item Name="ar.onnx" Type="Document" URL="../data/ar.onnx"/>
			<Item Name="coin_angle_detection.onnx" Type="Document" URL="../data/coin_angle_detection.onnx"/>
			<Item Name="coins_detection.onnx" Type="Document" URL="../data/coins_detection.onnx"/>
		</Item>
		<Item Name="Test EfficientNet_B4.vi" Type="VI" URL="../Test EfficientNet_B4.vi"/>
		<Item Name="Test YOLO.vi" Type="VI" URL="../Test YOLO.vi"/>
		<Item Name="image.lvlib" Type="Library" URL="../image_vi/image.lvlib"/>
		<Item Name="libonnx_wrapper.lvlib" Type="Library" URL="../libonnx_wrapper/libonnx_wrapper.lvlib"/>
		<Item Name="process_Angle.lvlib" Type="Library" URL="../process_Angle/process_Angle.lvlib"/>
		<Item Name="process_EfficientNet_B4.lvlib" Type="Library" URL="../process_EfficientNet_B4/process_EfficientNet_B4.lvlib"/>
		<Item Name="process_YOLOv8.lvlib" Type="Library" URL="../process_YOLOv8/process_YOLOv8.lvlib"/>
		<Item Name="Dependencies" Type="Dependencies"/>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
