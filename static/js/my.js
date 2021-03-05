$(document).ready(function () {
    $("#ApplicantIDInput").bind('input propertychange', 'none', function () {
        var text = $(this).val();
        if (text.length != 0) {
            $.ajax({
                type: "GET",
                url: "http://localhost:8080/FileManage/StaffIDQuery?",
                dataType: "json",
                data: {
                    StaffID: text,
                    power: "Applicant"
                },
                async: false,
                success: function (result) {
                    if (result && result.code == 0) {
                        console.log("success: " + JSON.stringify(result));
                        $('.info1')[0].style.display = "block";
                        $('.info2')[0].style.display = "none";
                        $('#ApplicantID')[0].value = result.data.staffID;
                        $('#ApplicantName')[0].value = result.data.staffName;
                        $('#ApplicantPhoneNumber')[0].value = result.data.staffPhone;
                        $('#DepartmentID')[0].value = result.data.departmentID;
                        $('#DepartmentName')[0].value = result.data.departmentName;
                    } else {
                        $('.info1')[0].style.display = "none";
                        $('.info2')[0].style.display = "block";
                        $('#ApplicantID')[0].value = "";
                        $('#ApplicantName')[0].value = "";
                        $('#ApplicantPhoneNumber')[0].value = "";
                        $('#DepartmentID')[0].value = "";
                        $('#DepartmentName')[0].value = "";
                    }

                },
                error: function (result) {
                    console.log("error: " + JSON.stringify(result));
                }
            })
        } else {
            $('.info1')[0].style.display = "none";
            $('.info2')[0].style.display = "none";
            $('#ApplicantID')[0].value = "";
            $('#ApplicantName')[0].value = "";
            $('#ApplicantPhoneNumber')[0].value = "";
            $('#DepartmentID')[0].value = "";
            $('#DepartmentName')[0].value = "";
        }
    });
    $("#ApplicantionIDInput").bind('input propertychange', 'none', function () {
        var text = $(this).val();
        if (text.length != 0) {
            $.ajax({
                type: "GET",
                url: "http://localhost:8080/FileManage/api/ApplicationInfo?",
                dataType: "json",
                data: {
                    loginid: "g71",
                    power: "Applicant",
                    ApplicationID: text,
                    token: sessionStorage.getItem("token")
                },
                async: false,
                success: function (result) {
                    if (result && result.code == 0) {
                        console.log("success: " + JSON.stringify(result));
                        $('.info3')[0].style.display = "none";
                        $('.info4')[0].style.display = "block";
                        $('#ApplicationID')[0].value = "";
                    } else if (result.code == 20001) {
                        $('.info3')[0].style.display = "none";
                        $('.info4')[0].style.display = "none";
                        $('#ApplicationID')[0].value = "";
                    } else if (result.code) {
                        $('.info3')[0].style.display = "block";
                        $('.info4')[0].style.display = "none";
                        $('#ApplicationID')[0].value = text;
                    }

                },
                error: function (result) {
                    console.log("error: " + JSON.stringify(result));
                }
            })
        } else {
            $('.info3')[0].style.display = "none";
            $('.info4')[0].style.display = "none";
            $('#ApplicationID')[0].value = "";
        }
    });
    $(".RecordType_Experiment").click(function () {
        if ($(this).is(":checked")) {
            $('#ExperimentIDLine')[0].style.display = "block";
        } else {
            $('#ExperimentIDLine')[0].style.display = "none";
        }
    });
    $(".RecordType_Operation").click(function () {
        if ($(this).is(":checked")) {
            $('#OperationIDLine')[0].style.display = "block";
        } else {
            $('#OperationIDLine')[0].style.display = "none";
        }
    });
    $(".RecordType_Rollin").click(function () {
        if ($(this).is(":checked")) {
            $('#RollinIDLine')[0].style.display = "block";
        } else {
            $('#RollinIDLine')[0].style.display = "none";
        }
    });
    $(".RecordType_Rollout").click(function () {
        if ($(this).is(":checked")) {
            $('#RolloutIDLine')[0].style.display = "block";
        } else {
            $('#RolloutIDLine')[0].style.display = "none";
        }
    });
    $(".RecordType_Inspect").click(function () {
        if ($(this).is(":checked")) {
            $('#InspectIDLine')[0].style.display = "block";
        } else {
            $('#InspectIDLine')[0].style.display = "none";
        }
    });
    $(".RecordType_Discussion").click(function () {
        if ($(this).is(":checked")) {
            $('#DiscussionIDLine')[0].style.display = "block";
        } else {
            $('#DiscussionIDLine')[0].style.display = "none";
        }
    });
});